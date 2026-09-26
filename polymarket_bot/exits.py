"""Exits: take profit, stop loss, and settlement.

Without this module the bot only ever opened positions, so every slot it owned
stayed owned and the position cap became a permanent ceiling. Three ways out
are supported:

1. **Take profit** — sell the whole position into the resting bids once the
   net proceeds clear the entry cost by ``take_profit_pct``.
2. **Stop loss** — sell once the net liquidation value has fallen
   ``stop_loss_pct`` below the entry cost. This is the only thing that bounds
   the loss on a directional bet.
3. **Settlement** — a resolved market pays $1 per winning share, so the
   position is closed at its true terminal value with no order needed.

Selling walks the bid side, so a quoted profit is only real if the book can
absorb the size. Exits are therefore all-or-nothing per position: for a hedged
set, every leg must be fully executable, otherwise selling one leg alone would
break the hedge and leave naked directional risk. When the book is too thin the
position is simply held.

If an exit only partly fills — one leg fails, or a leg fills short — the shares
that did sell are booked and **the rest stay on the ledger**. Dropping the whole
position there would hide risk the bot still holds, understate open exposure,
and let it open new trades against headroom it does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .config import Config
from .fees import FeeModel
from .logging_setup import get_logger
from .models import ZERO, MarketGroup, OrderBook, quantize_down
from .portfolio import Portfolio, Position

log = get_logger("exits")


@dataclass(frozen=True)
class Liquidation:
    """What selling a position at the current book would actually return."""

    executable: bool
    shares: Decimal = ZERO
    gross_usd: Decimal = ZERO
    fee_usd: Decimal = ZERO
    net_usd: Decimal = ZERO
    reason: str = ""

    @property
    def pnl_usd(self) -> Decimal:
        return self.net_usd


@dataclass
class ExitDecision:
    """Whether to close a position now, and why."""

    should_exit: bool
    reason: str = ""
    proceeds_usd: Decimal = ZERO
    pnl_usd: Decimal = ZERO
    detail: str = ""


@dataclass
class SettlementResult:
    """A position closed by market resolution."""

    position: Position
    payout_usd: Decimal
    pnl_usd: Decimal
    winning: bool


class ExitEngine:
    """Decide when to close positions and settle resolved ones."""

    def __init__(self, config: Config, broker: object, portfolio: Portfolio, scanner: object = None):
        self.config = config
        self.broker = broker
        self.portfolio = portfolio
        self.scanner = scanner
        self.exits_taken = 0
        self.stops_taken = 0
        self.settlements = 0
        # Filled by close(): the PnL actually booked, and any shares left on the
        # ledger when an exit only partly filled.
        self.last_pnl: Decimal = ZERO
        self.last_remaining: Decimal = ZERO

    # ------------------------------------------------------------- valuation
    def liquidate(self, position: Position, group: MarketGroup, fee: FeeModel) -> Liquidation:
        """Value an immediate exit, requiring every leg to be fully sellable."""
        if not position.legs:
            return Liquidation(False, reason="no leg detail recorded")

        books: dict[str, OrderBook] = {}
        for outcome in group.outcomes:
            if outcome.book is not None:
                books[outcome.token_id] = outcome.book

        gross = ZERO
        fee_total = ZERO
        sets = position.guaranteed_sets
        if sets <= 0:
            return Liquidation(False, reason="no shares held")

        for leg in position.legs:
            book = books.get(leg.token_id)
            if book is None or not book.bids:
                return Liquidation(False, reason=f"no bid for {leg.outcome_name}")
            # For a hedged set every leg must be sold at the same size, so the
            # smallest sellable leg caps the whole exit.
            sell_shares = min(leg.shares, sets) if position.hedged else leg.shares
            estimate = book.proceeds_to_sell(sell_shares)
            if estimate.shares < sell_shares:
                return Liquidation(
                    False,
                    reason=f"thin book for {leg.outcome_name}: "
                    f"{estimate.shares:.2f}/{sell_shares:.2f} sellable",
                )
            if estimate.usd <= 0:
                return Liquidation(False, reason=f"zero bid value for {leg.outcome_name}")
            gross += estimate.usd
            avg_price = estimate.usd / sell_shares if sell_shares > 0 else ZERO
            fee_total += fee.sell_fee(sell_shares, avg_price)

        net = gross - fee_total
        return Liquidation(True, shares=sets if position.hedged else position.shares,
                           gross_usd=gross, fee_usd=fee_total, net_usd=net)

    def evaluate(self, position: Position, group: MarketGroup, fee: FeeModel) -> ExitDecision:
        """Apply take-profit and stop-loss thresholds to one position."""
        if position.cost_usd <= 0:
            return ExitDecision(False)

        liq = self.liquidate(position, group, fee)
        if not liq.executable:
            return ExitDecision(False, detail=liq.reason)

        profit_pct = (liq.net_usd - position.cost_usd) / position.cost_usd

        if profit_pct >= Decimal(str(self.config.take_profit_pct)):
            return ExitDecision(
                True,
                reason="take_profit",
                proceeds_usd=liq.net_usd,
                pnl_usd=liq.net_usd - position.cost_usd,
                detail=f"net {profit_pct * 100:.2f}% vs target {self.config.take_profit_pct * 100:.2f}%",
            )
        if profit_pct <= -Decimal(str(self.config.stop_loss_pct)):
            return ExitDecision(
                True,
                reason="stop_loss",
                proceeds_usd=liq.net_usd,
                pnl_usd=liq.net_usd - position.cost_usd,
                detail=f"net {profit_pct * 100:.2f}% vs stop -{self.config.stop_loss_pct * 100:.2f}%",
            )
        return ExitDecision(False, detail=f"net {profit_pct * 100:.2f}% within band")

    # -------------------------------------------------------------- execution
    def close(self, position: Position, group: MarketGroup, decision: ExitDecision) -> bool:
        """Submit the sell orders for a position and book the result."""
        sets = position.guaranteed_sets
        proceeds = ZERO
        sold: dict[str, Decimal] = {}
        all_ok = True

        for leg in position.legs:
            sell_shares = min(leg.shares, sets) if position.hedged else leg.shares
            book = self._book_for(group, leg.token_id)
            price = self._limit_price(book, leg.entry_price)
            if price <= 0:
                all_ok = False
                log.warning("no bid to exit %s (%s)", position.title[:40], leg.outcome_name)
                break
            # Each leg exits on its own market's venue settings. A cross-market
            # position holds legs in two markets, so the position-wide tick
            # size would be wrong for one of them.
            tick = leg.tick_size if leg.tick_size > 0 else group.tick_size
            result = self.broker.sell(
                token_id=leg.token_id,
                shares=sell_shares,
                price=price,
                tick_size=tick,
                neg_risk=leg.neg_risk,
                order_type=self.config.live_order_type,
            )
            if result.ok and result.filled_shares > 0:
                sold[leg.token_id] = sold.get(leg.token_id, ZERO) + result.filled_shares
                proceeds += result.filled_usd
                if result.filled_shares < sell_shares:
                    all_ok = False
                    log.warning(
                        "exit leg partly filled for %s (%s): %s/%s shares",
                        position.title[:40],
                        leg.outcome_name,
                        result.filled_shares,
                        sell_shares,
                    )
                    break
            else:
                all_ok = False
                log.warning("exit leg failed for %s: %s", position.title[:40], result.error)
                break

        if not sold:
            return False

        pnl, remaining = self.portfolio.reduce_position(
            position, sold, proceeds, reason=decision.reason
        )

        if not all_ok:
            # Whatever did not sell is still on the ledger. For a hedged set
            # that means the hedge is broken and the leftover is naked risk,
            # which is worth shouting about rather than rounding away.
            log.error(
                "EXIT INCOMPLETE for %s — %.2f shares still held at risk, manual review required",
                position.title[:60],
                remaining,
            )

        if decision.reason == "stop_loss":
            self.stops_taken += 1
        else:
            self.exits_taken += 1
        self.last_pnl = pnl
        self.last_remaining = remaining
        return True

    @staticmethod
    def _book_for(group: MarketGroup, token_id: str) -> OrderBook | None:
        for outcome in group.outcomes:
            if outcome.token_id == token_id:
                return outcome.book
        return None

    @staticmethod
    def _limit_price(book: OrderBook | None, entry_price: Decimal) -> Decimal:
        """Worst price we accept: the best bid, never below a tick of value."""
        if book is None or not book.bids:
            return ZERO
        return book.bids[0].price

    # ------------------------------------------------------------- settlement
    def settlement_payout(self, position: Position, winning_token_ids: tuple[str, ...]) -> Decimal:
        """Terminal value of a resolved position: $1 per winning share."""
        return position.settlement_payout(winning_token_ids)

    def settle(self, position: Position, winning_token_ids: tuple[str, ...]) -> SettlementResult:
        payout = self.settlement_payout(position, winning_token_ids)
        pnl = self.portfolio.close_position(position, payout, reason="resolved")
        self.settlements += 1
        return SettlementResult(position, payout, pnl, winning=payout > 0)
