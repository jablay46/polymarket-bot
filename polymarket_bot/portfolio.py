"""Portfolio state: cash, open exposure, positions, and realized PnL.

The paper portfolio is a real ledger, not a counter. It tracks how much cash
each signal consumes, how much of that is still at risk, and how much is
locked in a hedged complete set. Live mode reads balances from the venue; the
ledger is only used for paper mode and for reporting.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from pathlib import Path

from .logging_setup import get_logger
from .models import ONE, ZERO, Signal

log = get_logger("portfolio")


@dataclass(frozen=True)
class PositionLeg:
    """One held leg: how many shares, and at what average entry price."""

    token_id: str
    outcome_name: str
    shares: Decimal
    entry_price: Decimal
    condition_id: str = ""

    @property
    def cost_usd(self) -> Decimal:
        return self.shares * self.entry_price


@dataclass
class Position:
    """The legs bought together for one signal.

    ``hedged`` is True only for a complete set, where the payout at resolution
    is a known $1 per set. Directional positions (the fade strategy) carry an
    *assumed* exit price instead, so their profit is not comparable.
    """

    signal_kind: str
    group_id: str
    title: str
    token_ids: tuple[str, ...]
    shares: Decimal
    cost_usd: Decimal
    opened_at: float
    edge_per_set: Decimal
    expected_payout_usd: Decimal
    hedged: bool = True
    legs: tuple[PositionLeg, ...] = ()
    condition_id: str = ""
    theme: str = ""
    tick_size: Decimal = Decimal("0.01")
    neg_risk: bool = False
    payout_per_set: Decimal = ONE

    @property
    def expected_profit_usd(self) -> Decimal:
        return self.expected_payout_usd - self.cost_usd

    @property
    def guaranteed_sets(self) -> Decimal:
        """Complete sets actually held: the smallest leg, not the largest.

        A pair of legs with unequal fills only locks in ``min(shares)``; the
        excess on the bigger leg is unhedged directional risk.
        """
        if not self.legs:
            return self.shares
        return min((leg.shares for leg in self.legs), default=self.shares)

    def leg_for(self, token_id: str) -> PositionLeg | None:
        for leg in self.legs:
            if leg.token_id == token_id:
                return leg
        return None

    def settlement_payout(self, winning_token_ids: tuple[str, ...]) -> Decimal:
        """Cash returned when the market settles.

        Every held share of a winning token pays $1; losing tokens pay nothing.
        """
        winners = set(winning_token_ids)
        payout = ZERO
        for leg in self.legs:
            if leg.token_id in winners:
                payout += leg.shares
        return payout


@dataclass
class Portfolio:
    """Cash and position ledger for paper trading."""

    starting_balance: Decimal = Decimal("1000")
    cash: Decimal = field(init=False, default=ZERO)
    realized_pnl: Decimal = Decimal(0)
    positions: list[Position] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.starting_balance = Decimal(str(self.starting_balance))
        if self.cash == 0:
            self.cash = self.starting_balance

    # ------------------------------------------------------------ accounting
    @property
    def open_exposure(self) -> Decimal:
        """Cash currently committed to open positions."""
        return sum((p.cost_usd for p in self.positions), ZERO)

    @property
    def expected_payout(self) -> Decimal:
        """Certain payout, from hedged complete sets only."""
        return sum((p.expected_payout_usd for p in self.positions if p.hedged), ZERO)

    @property
    def expected_profit(self) -> Decimal:
        """Certain profit locked in by hedged complete sets."""
        return sum((p.expected_profit_usd for p in self.positions if p.hedged), ZERO)

    @property
    def directional_exposure(self) -> Decimal:
        """Cash at risk in unhedged, model-driven positions."""
        return sum((p.cost_usd for p in self.positions if not p.hedged), ZERO)

    @property
    def assumed_profit(self) -> Decimal:
        """Profit that depends on a directional position's exit assumption."""
        return sum((p.expected_profit_usd for p in self.positions if not p.hedged), ZERO)

    @property
    def equity(self) -> Decimal:
        return self.cash + self.open_exposure

    @property
    def open_positions(self) -> int:
        return len(self.positions)

    def available_cash(self) -> Decimal:
        return self.cash

    # -------------------------------------------------------------- mutation
    def holds(self, group_id: str) -> bool:
        """True if a position is already open on this market.

        Keyed on ``group_id`` (the condition id), which is stable across
        cycles, so a market can never be bought twice.
        """
        if not group_id:
            return False
        with self._lock:
            return any(p.group_id == group_id for p in self.positions)

    def theme_exposure(self, theme: str) -> Decimal:
        """Open exposure across every position sharing a correlation theme."""
        if not theme:
            return ZERO
        with self._lock:
            return sum((p.cost_usd for p in self.positions if p.theme == theme), ZERO)

    def open_position(
        self,
        signal: Signal,
        cost_usd: Decimal,
        *,
        fills: tuple[tuple, ...] = (),
        condition_id: str = "",
        theme: str = "",
        tick_size: Decimal = Decimal("0.01"),
        neg_risk: bool = False,
    ) -> Position:
        """Record a filled position and debit the cash it consumed.

        ``cost_usd`` is the cash actually spent. ``fills`` carries the real
        per-leg outcome as ``(token_id, outcome_name, shares, avg_price)`` or
        ``(token_id, outcome_name, shares, avg_price, condition_id)``; when
        omitted the signal's planned legs are assumed to have filled in full,
        which keeps the ledger honest for callers that do not trade live.
        """
        cost_usd = Decimal(str(cost_usd))
        with self._lock:
            if cost_usd > self.cash:
                raise ValueError(f"insufficient cash: need {cost_usd}, have {self.cash}")
            self.cash -= cost_usd

            if fills:
                legs = tuple(
                    PositionLeg(
                        str(f[0]),
                        str(f[1]),
                        Decimal(str(f[2])),
                        Decimal(str(f[3])),
                        str(f[4]) if len(f) > 4 else "",
                    )
                    for f in fills
                    if Decimal(str(f[2])) > 0
                )
                held_shares = sum((leg.shares for leg in legs), ZERO)
            else:
                legs = tuple(
                    PositionLeg(leg.token_id, leg.outcome_name, leg.shares, leg.price)
                    for leg in signal.legs
                )
                held_shares = signal.max_sets

            # Payout is based on shares actually held, not the planned size: a
            # partial fill must not be valued as if the whole order went through.
            # A hedged set pays $1 per complete set; a directional position
            # carries its assumed exit price instead.
            directional = bool(signal.metadata.get("directional", False))
            if directional:
                payout = held_shares * signal.payout_per_set
            else:
                payout = min((leg.shares for leg in legs), default=held_shares)
            position = Position(
                signal_kind=signal.kind,
                group_id=signal.group_id,
                title=signal.title,
                token_ids=tuple(leg.token_id for leg in legs),
                shares=held_shares,
                cost_usd=cost_usd,
                opened_at=time.time(),
                edge_per_set=signal.edge_per_set,
                expected_payout_usd=payout,
                hedged=not directional,
                legs=legs,
                condition_id=condition_id,
                theme=theme,
                tick_size=tick_size,
                neg_risk=neg_risk,
                payout_per_set=signal.payout_per_set,
            )
            self.positions.append(position)
            self.history.append(
                {
                    "event": "open",
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "kind": signal.kind,
                    "title": signal.title,
                    "group_id": signal.group_id,
                    "theme": theme,
                    "shares": str(position.shares),
                    "cost_usd": str(cost_usd),
                    "edge_per_set": str(signal.edge_per_set),
                    "expected_profit_usd": str(position.expected_profit_usd),
                }
            )
            return position

    def close_position(self, position: Position, proceeds_usd: Decimal, reason: str = "resolved") -> Decimal:
        """Close a position, crediting `proceeds_usd` back to cash."""
        with self._lock:
            proceeds = Decimal(str(proceeds_usd))
            if position in self.positions:
                self.positions.remove(position)
            self.cash += proceeds
            pnl = proceeds - position.cost_usd
            self.realized_pnl += pnl
            self.history.append(
                {
                    "event": "close",
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "kind": position.signal_kind,
                    "title": position.title,
                    "group_id": position.group_id,
                    "theme": position.theme,
                    "proceeds_usd": str(proceeds),
                    "cost_usd": str(position.cost_usd),
                    "pnl_usd": str(pnl),
                    "reason": reason,
                }
            )
            return pnl

    def reduce_position(
        self,
        position: Position,
        sold_shares: dict[str, Decimal],
        proceeds_usd: Decimal,
        reason: str = "partial",
    ) -> tuple[Decimal, Decimal]:
        """Book an exit for the shares that actually sold.

        Returns ``(pnl, remaining_shares)``. Whatever did not sell stays on the
        ledger: a position whose exit was only partly filled still carries real
        risk, so dropping it from ``positions`` would understate open exposure
        and let the bot open new trades against risk it already holds. The
        position is only removed once every share is gone.
        """
        with self._lock:
            proceeds = Decimal(str(proceeds_usd))
            sold_cost = ZERO
            remaining_legs: list[PositionLeg] = []
            sold_total = ZERO
            for leg in position.legs:
                sold = min(Decimal(str(sold_shares.get(leg.token_id, ZERO))), leg.shares)
                sold_total += sold
                sold_cost += sold * leg.entry_price
                left = leg.shares - sold
                if left > 0:
                    remaining_legs.append(replace(leg, shares=left))

            self.cash += proceeds
            pnl = proceeds - sold_cost
            self.realized_pnl += pnl
            remaining_shares = sum((leg.shares for leg in remaining_legs), ZERO)

            if remaining_legs:
                # Revalue what is left so exposure and payout track reality.
                hedge_intact = position.hedged and len(remaining_legs) == len(position.legs)
                if hedge_intact:
                    payout = min((leg.shares for leg in remaining_legs), default=ZERO) * position.payout_per_set
                elif position.hedged:
                    # A leg sold out entirely, so the hedge is broken and the
                    # leftover is naked. Value it at cost rather than claiming
                    # a guaranteed payout it no longer has.
                    left_cost = sum((leg.cost_usd for leg in remaining_legs), ZERO)
                    payout = left_cost
                else:
                    payout = remaining_shares * position.payout_per_set
                reduced = replace(
                    position,
                    legs=tuple(remaining_legs),
                    token_ids=tuple(leg.token_id for leg in remaining_legs),
                    shares=remaining_shares,
                    cost_usd=position.cost_usd - sold_cost,
                    expected_payout_usd=payout,
                    hedged=hedge_intact,
                )
                for i, held in enumerate(self.positions):
                    if held is position:
                        self.positions[i] = reduced
                        break
            else:
                for i, held in enumerate(self.positions):
                    if held is position:
                        del self.positions[i]
                        break

            self.history.append(
                {
                    "event": "close" if not remaining_legs else "reduce",
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "kind": position.signal_kind,
                    "title": position.title,
                    "group_id": position.group_id,
                    "theme": position.theme,
                    "proceeds_usd": str(proceeds),
                    "sold_shares": str(sold_total),
                    "remaining_shares": str(remaining_shares),
                    "cost_usd": str(sold_cost),
                    "pnl_usd": str(pnl),
                    "reason": reason,
                }
            )
            return pnl, remaining_shares

    # ------------------------------------------------------------ persistence
    def snapshot(self) -> dict:
        return {
            "starting_balance": str(self.starting_balance),
            "cash": str(self.cash),
            "realized_pnl": str(self.realized_pnl),
            "open_exposure": str(self.open_exposure),
            "hedged_exposure": str(self.open_exposure - self.directional_exposure),
            "directional_exposure": str(self.directional_exposure),
            "expected_payout": str(self.expected_payout),
            "expected_profit": str(self.expected_profit),
            "assumed_profit": str(self.assumed_profit),
            "open_positions": self.open_positions,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "snapshot": self.snapshot(),
            "positions": [
                {
                    **asdict(p),
                    "shares": str(p.shares),
                    "cost_usd": str(p.cost_usd),
                    "edge_per_set": str(p.edge_per_set),
                    "expected_payout_usd": str(p.expected_payout_usd),
                    "tick_size": str(p.tick_size),
                    "payout_per_set": str(p.payout_per_set),
                    "token_ids": list(p.token_ids),
                    "legs": [
                        {
                            "token_id": leg.token_id,
                            "outcome_name": leg.outcome_name,
                            "shares": str(leg.shares),
                            "entry_price": str(leg.entry_price),
                            "condition_id": leg.condition_id,
                        }
                        for leg in p.legs
                    ],
                }
                for p in self.positions
            ],
            "history": self.history[-500:],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def summary(self) -> str:
        return (
            f"cash=${self.cash:.2f} exposure=${self.open_exposure:.2f} "
            f"equity=${self.equity:.2f} open={self.open_positions} "
            f"realized=${self.realized_pnl:.2f} "
            f"hedged_profit=${self.expected_profit:.2f} "
            f"directional=${self.directional_exposure:.2f} (assumed_profit=${self.assumed_profit:.2f})"
        )
