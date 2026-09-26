"""Cross-market logical arbitrage on threshold ladders.

Kept deliberately separate from ``strategies.py`` and from
:class:`~polymarket_bot.strategies.StrategyEngine`. The other three
strategies price a risk-free position from facts the CLOB itself guarantees
(a binary market's Yes+No always partition $1; a neg-risk event's outcomes
are enforced mutually exclusive by the protocol). This one prices a
position from a *relation this code guessed at* — see
``polymarket_bot.relations`` for why that guess can be wrong in a way a
losing bet is not. Signals from here therefore carry
``metadata["unverified_relation"] = True`` and are never auto-executed by
:meth:`~polymarket_bot.engine.TradingEngine.run_cycle`; wiring that up is a
deliberate choice left to whoever reviews and enables it, not a default.

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
from .models import ONE, ZERO, Leg, OrderBook, Signal, quantize_down
from .relations import ThresholdCandidate, extract_direction, extract_threshold

MAX_LEVELS = 40


def candidate_from_market(market) -> ThresholdCandidate | None:
    """Build a ladder candidate from a scanned binary market, or None.

    Returns None unless the market is binary (one Yes, one No) and its
    question carries a readable threshold. ``direction`` comes from the
    question wording; see :func:`polymarket_bot.relations.extract_direction`.
    """
    if len(market.token_ids) != 2:
        return None
    threshold = extract_threshold(market.question)
    if threshold is None:
        return None
    return ThresholdCandidate(
        market_id=market.market_id,
        condition_id=market.condition_id,
        question=market.question,
        threshold=threshold,
        yes_token_id=market.token_ids[0],
        no_token_id=market.token_ids[1],
        direction=extract_direction(market.question),
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
            if book.top_ask_size < Decimal(str(self.config.arb_min_top_size)):
                return None

        max_gross = Decimal(str(self.config.arb_max_edge))
        min_edge = Decimal(str(self.config.arb_min_edge))

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
            return (gross, gross - fee_per_set, cost, usd, yes_fill, no_fill)

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
        gross, net, cost, usd, yes_fill, no_fill = result
        if net < min_edge:
            return None

        legs = (
            Leg(
                implied.yes_token_id, f"YES: {implied.question[:40]}", yes_fill.avg_price, capacity,
                yes_fill.usd, touch_price=yes_book.best_ask,
            ),
            Leg(
                implying.no_token_id, f"NO: {implying.question[:40]}", no_fill.avg_price, capacity,
                no_fill.usd, touch_price=no_book.best_ask,
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
                "gross_edge": str(gross),
                "direction": lower.direction,
                "implied_market_id": implied.market_id,
                "implying_market_id": implying.market_id,
                "lower_market_id": lower.market_id,
                "higher_market_id": higher.market_id,
                "lower_threshold": str(lower.threshold),
                "higher_threshold": str(higher.threshold),
                "warning": (
                    "relation detected by keyword pattern, not verified against the "
                    "markets' actual resolution rules — confirm by hand before trading"
                ),
            },
        )

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
