"""Cross-market logical arbitrage on threshold ladders.

Kept deliberately separate from ``strategies.py`` and from
:class:`~polymarket_bot.strategies.StrategyEngine`. The other three
strategies price a risk-free position from facts the CLOB itself guarantees
(a binary market's Yes+No always partition $1; a neg-risk event's outcomes
are enforced mutually exclusive by the protocol). This one prices a
position from a *relation this code guessed at* — see
``polymarket_bot.relations`` for why that guess can be wrong in a way a
losing bet is not. Signals from here therefore carry
``metadata["unverified_relation"] = True``. In paper mode the engine will
price and fill them so an operator can see the relation behave; in live
mode every pair must be named in
``POLYMARKET_BOT_CROSS_MARKET_CONFIRMED_PAIRS`` or it is refused before it
is priced. That list is empty by default, so enabling the strategy alone
never puts a guessed relation on the real book.

The trade, once a genuine relation is confirmed
--------------------------------------------------
The relation on a ladder is an implication, and the safe side is always the
same shape: **buy Yes on the implied market and No on the implying one.**

* "above" ladder, ascending thresholds ``t_low < t_high``: reaching the
  higher threshold implies reaching the lower one, so ``above(t_high) =>
  above(t_low)``. Buy **Yes(low) + No(high)**.
* "below" ladder, ascending thresholds ``t_low < t_high``: being below the
  lower threshold implies being below the higher one, so ``below(t_low) =>
  below(t_high)``. Buy **Yes(high) + No(low)** — the opposite pair.

Both pay $1 in every state the relation allows, so the two must not be
confused. Check the "above" case (``A = above(t_high)``, ``B = above(t_low)``,
holding Yes(B) + No(A)):

* A true, B true: No(A) $0 + Yes(B) $1 -> $1.
* A false, B true: No(A) $1 + Yes(B) $1 -> $2.
* A false, B false: No(A) $1 + Yes(B) $0 -> $1.
* A true, B false: impossible under the relation; if it happens the relation
  is false, which is exactly the risk flagged above.

Cost is ``Yes(implied).ask + No(implying).ask``, read straight off each leg's
own book, exactly as :class:`~polymarket_bot.strategies.SetArbitrageStrategy`
prices a same-market Yes+No pair — the arithmetic is identical; only the
source of the "these two legs cover every outcome" guarantee differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from .config import Config
from .fees import FeeModel
from .logging_setup import get_logger
from .models import ONE, ZERO, Leg, OrderBook, Signal, quantize_down
from .relations import ThresholdCandidate, extract_direction, extract_threshold

MAX_LEVELS = 40

log = get_logger("cross_market")


def _resolve_yes_index(market) -> int:
    """Index of the Yes token in a two-outcome market.

    Gamma does not guarantee that ``clobTokenIds`` lists Yes first, so
    ``token_ids[0]`` is not safe to treat as Yes. Prefer the market's own
    resolution (``MarketInfo.yes_index``, matched by outcome name), then the
    outcome literally named "yes", then fall back to 0 — a two-outcome market
    with no Yes/No labels still partitions $1, so either side is a valid
    reference there.
    """
    index = getattr(market, "yes_index", None)
    if index in (0, 1):
        return index
    for i, name in enumerate(getattr(market, "outcomes", None) or ()):
        if str(name).strip().lower() == "yes":
            return i
    return 0


def candidate_from_market(market) -> ThresholdCandidate | None:
    """Build a ladder candidate from a scanned binary market, or None.

    Returns None unless the market is binary (one Yes, one No) and its
    question carries a readable threshold. ``direction`` comes from the
    question wording; see :func:`polymarket_bot.relations.extract_direction`.
    Venue settings and size filters travel with the candidate so a pair can be
    checked against each market's own limits.
    """
    if len(market.token_ids) != 2:
        return None
    threshold = extract_threshold(market.question)
    if threshold is None:
        return None
    yes_index = _resolve_yes_index(market)
    no_index = 1 - yes_index
    return ThresholdCandidate(
        market_id=market.market_id,
        condition_id=market.condition_id,
        question=market.question,
        threshold=threshold,
        yes_token_id=market.token_ids[yes_index],
        no_token_id=market.token_ids[no_index],
        direction=extract_direction(market.question),
        tick_size=market.tick_size,
        neg_risk=bool(market.neg_risk),
        volume_24h=Decimal(str(market.volume_24h or 0)),
        liquidity=Decimal(str(market.liquidity or 0)),
        seconds_to_end=market.seconds_to_end,
        fees_enabled=bool(getattr(market, "fees_enabled", False)),
        fee_type=getattr(market, "fee_type", None),
    )


@dataclass
class CrossMarketArbitrageStrategy:
    """Price a No(lower)/Yes(higher) pair on a candidate threshold ladder."""

    config: Config

    def evaluate_pair(
        self,
        lower: ThresholdCandidate,
        higher: ThresholdCandidate,
        books: Mapping[str, OrderBook],
        fee_lower: FeeModel,
        fee_higher: FeeModel,
    ) -> Signal | None:
        """Price the safe side of the implication ``lower``/``higher`` carry.

        ``books`` maps token id to book. The strategy picks which two of the
        four tokens it needs from the ladder direction — see the module
        docstring. Both legs are read at the ask, since both are buys.
        """
        if lower.direction != higher.direction:
            # Different implication directions are not ordered against each
            # other, so there is no relation to price.
            return None

        if not self._passes_market_filters(lower, higher):
            return None
        if self.config.is_live and not self._pair_confirmed(lower, higher):
            log.warning(
                "cross-market pair %s/%s not in the confirmed list; skipping in live mode "
                "(set POLYMARKET_BOT_CROSS_MARKET_CONFIRMED_PAIRS after checking both "
                "markets' resolution rules)",
                lower.market_id[:16],
                higher.market_id[:16],
            )
            return None

        # "above": above(higher) implies above(lower) -> Yes(lower) + No(higher).
        # "below": below(lower) implies below(higher) -> Yes(higher) + No(lower).
        implied, implying = (
            (lower, higher) if lower.direction == "above" else (higher, lower)
        )
        yes_book = books.get(implied.yes_token_id)
        no_book = books.get(implying.no_token_id)
        fee_yes = fee_lower if implied is lower else fee_higher
        fee_no = fee_lower if implying is lower else fee_higher

        if yes_book is None or no_book is None:
            return None
        if not yes_book.has_two_sided_market() or not no_book.has_two_sided_market():
            return None

        max_spread = Decimal(str(self.config.arb_max_spread))
        for book in (yes_book, no_book):
            spread = book.spread
            if spread is None or spread > max_spread:
                return None
            if book.top_ask_size < Decimal(str(self.config.cross_market_min_top_size)):
                return None

        max_gross = Decimal(str(self.config.cross_market_max_edge))
        min_edge = Decimal(str(self.config.cross_market_min_edge))

        yes_depth = sum((lvl.size for lvl in yes_book.asks[:MAX_LEVELS]), ZERO)
        no_depth = sum((lvl.size for lvl in no_book.asks[:MAX_LEVELS]), ZERO)
        capacity = quantize_down(min(yes_depth, no_depth), ONE)
        if capacity <= 0:
            return None

        def evaluate(sets: Decimal):
            if sets <= 0:
                return None
            yes_fill = yes_book.cost_to_buy(sets, price_limit=ONE)
            no_fill = no_book.cost_to_buy(sets, price_limit=ONE)
            if not yes_fill.complete or not no_fill.complete:
                return None
            usd = yes_fill.usd + no_fill.usd
            cost = usd / sets
            gross = ONE - cost
            if gross > max_gross:
                return None
            fee_per_set = (
                fee_yes.buy_fee(sets, yes_fill.avg_price) + fee_no.buy_fee(sets, no_fill.avg_price)
            ) / sets
            return (gross, gross - fee_per_set, cost, usd, yes_fill, no_fill, fee_per_set)

        def profitable(sets: Decimal) -> bool:
            result = evaluate(sets)
            return result is not None and result[1] >= min_edge

        if not profitable(capacity):
            low, high = ZERO, capacity
            for _ in range(48):
                if high - low <= ONE:
                    break
                mid = quantize_down((low + high) / 2, ONE)
                if mid <= low:
                    break
                if profitable(mid):
                    low = mid
                else:
                    high = mid
            capacity = quantize_down(low, ONE)
            if capacity <= 0:
                return None

        result = evaluate(capacity)
        if result is None:
            return None
        gross, net, cost, usd, yes_fill, no_fill, fee_per_set = result
        if net < min_edge:
            return None

        legs = (
            Leg(
                implied.yes_token_id, f"YES: {implied.question[:40]}", yes_fill.avg_price, capacity,
                yes_fill.usd, touch_price=yes_book.best_ask,
                condition_id=implied.condition_id, tick_size=implied.tick_size,
                neg_risk=implied.neg_risk,
            ),
            Leg(
                implying.no_token_id, f"NO: {implying.question[:40]}", no_fill.avg_price, capacity,
                no_fill.usd, touch_price=no_book.best_ask,
                condition_id=implying.condition_id, tick_size=implying.tick_size,
                neg_risk=implying.neg_risk,
            ),
        )
        return Signal(
            kind="cross_market_arbitrage",
            group_id=f"{lower.condition_id}:{higher.condition_id}",
            title=f"{implying.question[:36]} => {implied.question[:36]}",
            legs=legs,
            edge_per_set=net,
            cost_per_set=cost,
            payout_per_set=ONE,
            confidence=0.0,  # deliberately not scored like the verified strategies — see module docstring
            expected_profit_usd=capacity * net,
            max_sets=capacity,
            metadata={
                "unverified_relation": True,
                "directional": False,
                "gross_edge": str(gross),
                "fee_per_set": str(fee_per_set),
                "direction": lower.direction,
                "implied_market_id": implied.market_id,
                "implying_market_id": implying.market_id,
                "lower_market_id": lower.market_id,
                "higher_market_id": higher.market_id,
                "lower_threshold": str(lower.threshold),
                "higher_threshold": str(higher.threshold),
                "confirmed_pair": self._pair_confirmed(lower, higher),
                # Execution records one position spanning two markets, so it
                # needs each leg's condition id rather than one group id.
                "cross_market": True,
                "leg_condition_ids": (implied.condition_id, implying.condition_id),
                "warning": (
                    "relation detected by keyword pattern, not verified against the "
                    "markets' actual resolution rules — confirm by hand before trading"
                ),
            },
        )

    # ------------------------------------------------------------ live gating
    def _pair_key(self, lower: ThresholdCandidate, higher: ThresholdCandidate) -> tuple:
        return tuple(sorted((lower.condition_id, higher.condition_id)))

    def _pair_confirmed(self, lower: ThresholdCandidate, higher: ThresholdCandidate) -> bool:
        return self._pair_key(lower, higher) in self.config.confirmed_cross_market_pairs

    def _passes_market_filters(
        self, lower: ThresholdCandidate, higher: ThresholdCandidate
    ) -> bool:
        """Both markets must clear the volume, liquidity, and time floors.

        A ladder leg in a dead market cannot be exited, and the two legs are
        only mutually coverable if both can actually be traded.
        """
        min_volume = Decimal(str(self.config.cross_market_min_volume_24h))
        min_liquidity = Decimal(str(self.config.cross_market_min_liquidity))
        min_seconds = self.config.cross_market_min_seconds_left
        for candidate in (lower, higher):
            if candidate.volume_24h < min_volume:
                return False
            if candidate.liquidity < min_liquidity:
                return False
            if candidate.seconds_to_end is not None and candidate.seconds_to_end < min_seconds:
                return False
        return True

    def find_signals(self, markets, scanner, fee_for) -> list[Signal]:
        """Scan markets for candidate ladders and price every adjacent pair.

        ``markets`` is a sequence of scanned binary markets, ``scanner`` a
        :class:`~polymarket_bot.data.MarketScanner` used to fetch books, and
        ``fee_for`` maps a market to its :class:`~polymarket_bot.fees.FeeModel`.
        Books are fetched only for tokens that belong to a ladder of 2+, so
        the extra request cost is paid only where a relation might exist.
        """
        from .relations import adjacent_pairs, group_threshold_ladders

        candidates = [c for c in (candidate_from_market(m) for m in markets) if c is not None]
        ladders = group_threshold_ladders(candidates)
        if not ladders:
            return []

        wanted = sorted({t for ladder in ladders.values() for c in ladder for t in (c.yes_token_id, c.no_token_id)})
        books = scanner.fetch_books(wanted)

        signals: list[Signal] = []
        for ladder in ladders.values():
            for low, high in adjacent_pairs(ladder):
                signal = self.evaluate_pair(
                    low,
                    high,
                    books,
                    fee_for(low),
                    fee_for(high),
                )
                if signal is not None:
                    signals.append(signal)
        return signals