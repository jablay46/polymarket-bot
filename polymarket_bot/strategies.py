"""Trading strategies.

Three strategies are provided, in decreasing order of reliability:

``SetArbitrageStrategy``
    Binary complete-set arbitrage. Buying Yes and No for less than one dollar
    locks a profit because exactly one side pays out at resolution. Sizing
    walks both books level by level, so the reported size and profit are
    constrained by real depth rather than a nominal top-of-book price.

``BasketArbitrageStrategy``
    The same idea generalized to a mutually exclusive outcome set (a
    Polymarket neg-risk event such as "EPL 2027 Champion"). Exactly one of the
    N outcomes resolves Yes, so buying every Yes for a combined price below
    one dollar is again risk-free. Depth is handled with a monotone search
    over the joint cost curve.

``FadeExtremeStrategy``
    Directional mean reversion on very liquid markets trading at extreme
    prices. This is *not* risk-free and is disabled by default. Its edge is an
    explicit heuristic, documented where it is computed.

All strategies are pure functions of market data plus fees. They perform no
I/O, which keeps them straightforward to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .config import Config
from .fees import FeeModel
from .logging_setup import get_logger
from .models import (
    ONE,
    ZERO,
    Leg,
    MarketGroup,
    Outcome,
    Signal,
    quantize_down,
)

log = get_logger("strategy")

# Cap how deep into the book we are willing to walk. Levels beyond this are
# usually stale quotes at absurd prices, not real arbitrage liquidity.
MAX_LEVELS = 40



def _confidence(edge_per_set: Decimal, target: Decimal, depth_usd: Decimal) -> float:
    """Blend edge size and available depth into a 0..1 confidence score."""
    if target <= 0:
        edge_score = Decimal(1)
    else:
        edge_score = min(Decimal(1), edge_per_set / (target * 2))
    depth_score = min(Decimal(1), depth_usd / Decimal(250))
    blended = (edge_score * Decimal("0.65")) + (depth_score * Decimal("0.35"))
    return round(float(max(Decimal("0.05"), min(Decimal("0.99"), blended))), 4)


def _capacity_shares(group: MarketGroup) -> Decimal:
    """Largest whole-share set the joint book can support."""
    return quantize_down(min(_depth_shares(o) for o in group.outcomes), ONE)


@dataclass
class SetArbitrageStrategy:
    """Binary complete-set arbitrage with depth-aware sizing."""

    config: Config

    def evaluate(self, group: MarketGroup, fee: FeeModel) -> Signal | None:
        if not group.is_binary or len(group.outcomes) != 2:
            return None
        yes, no = group.outcomes
        if yes.book is None or no.book is None:
            return None
        if not yes.book.has_two_sided_market() or not no.book.has_two_sided_market():
            return None

        # Book-integrity filters.
        max_spread = Decimal(str(self.config.arb_max_spread))
        for outcome in (yes, no):
            spread = outcome.book.spread
            if spread is None or spread > max_spread:
                return None
            if outcome.book.top_ask_size < Decimal(str(self.config.arb_min_top_size)):
                return None

        max_gross = Decimal(str(self.config.arb_max_edge))
        sets, gross_edge, net_edge, cost, usd = self._best_depth(yes, no, fee, max_gross)
        if sets <= 0 or net_edge < Decimal(str(self.config.arb_min_edge)):
            return None

        yes_fill = yes.book.cost_to_buy(sets, price_limit=ONE)
        no_fill = no.book.cost_to_buy(sets, price_limit=ONE)
        if not yes_fill.complete or not no_fill.complete:
            return None

        legs = (
            Leg(yes.token_id, yes.name, yes_fill.avg_price, sets, yes_fill.usd, touch_price=yes.book.best_ask),
            Leg(no.token_id, no.name, no_fill.avg_price, sets, no_fill.usd, touch_price=no.book.best_ask),
        )
        total_usd = yes_fill.usd + no_fill.usd
        profit = sets * net_edge
        return Signal(
            kind="set_arbitrage",
            group_id=group.group_id,
            title=group.title,
            legs=legs,
            edge_per_set=net_edge,
            cost_per_set=cost,
            payout_per_set=ONE,
            confidence=_confidence(net_edge, Decimal(str(self.config.arb_min_edge)), total_usd),
            expected_profit_usd=profit,
            max_sets=sets,
            metadata={
                "gross_edge": str(gross_edge),
                "fee_per_set": str(gross_edge - net_edge),
                "taker_rate": str(fee.taker_rate),
                "is_binary": True,
            },
        )

    def _best_depth(
        self,
        yes: Outcome,
        no: Outcome,
        fee: FeeModel,
        max_gross: Decimal,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
        """Find the largest set size with a profitable net edge.

        Cumulative depth is non-decreasing in set size on both legs, so the
        average cost per set is non-decreasing too. Profitability is therefore
        monotone: once a size stops being profitable, every larger size is
        worse. A binary search over sizes is exact and cheap.

        Returns ``(sets, gross_edge, net_edge, cost_per_set, usd)``; ``sets`` is
        zero when no profitable size exists.
        """
        min_edge = Decimal(str(self.config.arb_min_edge))
        yes_depth = sum((lvl.size for lvl in yes.book.asks[:MAX_LEVELS]), ZERO)
        no_depth = sum((lvl.size for lvl in no.book.asks[:MAX_LEVELS]), ZERO)
        # Book sizes are not integral in general, so work in whole shares.
        capacity = quantize_down(min(yes_depth, no_depth), ONE)
        if capacity <= 0:
            return (ZERO, ZERO, ZERO, ZERO, ZERO)

        def evaluate(sets: Decimal):
            if sets <= 0:
                return None
            y_fill = yes.book.cost_to_buy(sets, price_limit=ONE)
            n_fill = no.book.cost_to_buy(sets, price_limit=ONE)
            if not y_fill.complete or not n_fill.complete:
                return None
            usd = y_fill.usd + n_fill.usd
            cost = usd / sets
            gross = ONE - cost
            if gross > max_gross:
                return None
            fee_per_set = (fee.buy_fee(sets, y_fill.avg_price) + fee.buy_fee(sets, n_fill.avg_price)) / sets
            return (gross, gross - fee_per_set, cost, usd)

        def profitable(sets: Decimal) -> bool:
            result = evaluate(sets)
            return result is not None and result[1] >= min_edge

        if not profitable(capacity):
            # Capacity is where average cost is highest, so shrink to the
            # largest size that still clears the edge.
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
                return (ZERO, ZERO, ZERO, ZERO, ZERO)

        result = evaluate(capacity)
        if result is None:
            return (ZERO, ZERO, ZERO, ZERO, ZERO)
        gross, net, cost, usd = result
        return (capacity, gross, net, cost, usd)


@dataclass
class BasketArbitrageStrategy:
    """Neg-risk basket arbitrage with a monotone size search."""

    config: Config

    def evaluate(self, group: MarketGroup, fee: FeeModel) -> Signal | None:
        if group.is_binary or group.n_outcomes < self.config.basket_min_outcomes:
            return None
        if group.n_outcomes > self.config.basket_max_outcomes:
            return None
        if not group.with_books_present():
            return None
        for outcome in group.outcomes:
            if not outcome.book.has_two_sided_market():
                return None
            if outcome.book.top_ask_size < Decimal(str(self.config.basket_min_top_size)):
                return None

        ask_sum = group.ask_sum()
        if ask_sum is None or ask_sum <= 0:
            return None
        # Sanity: a mutually exclusive set must price near one.
        dislocation = abs(ask_sum - ONE)
        if dislocation > Decimal(str(self.config.basket_max_dislocation)):
            return None
        if ask_sum >= ONE:
            return None

        max_depth = _capacity_shares(group)
        if max_depth <= 0:
            return None

        sets = self._max_profitable_sets(group, fee, max_depth)
        if sets is None or sets <= 0:
            return None

        fills = [o.book.cost_to_buy(sets, price_limit=ONE) for o in group.outcomes]
        if any(not f.complete for f in fills):
            return None
        total_usd = sum((f.usd for f in fills), ZERO)
        cost_per_set = total_usd / sets
        gross_edge = ONE - cost_per_set
        if gross_edge > Decimal(str(self.config.basket_max_edge)):
            return None
        fee_per_set = sum(
            (fee.buy_fee(sets, f.avg_price) for f in fills), ZERO
        ) / sets
        net_edge = gross_edge - fee_per_set
        if net_edge < Decimal(str(self.config.basket_min_edge)):
            return None

        legs = tuple(
            Leg(o.token_id, o.name, f.avg_price, sets, f.usd, touch_price=o.book.best_ask)
            for o, f in zip(group.outcomes, fills)
        )
        return Signal(
            kind="basket_arbitrage",
            group_id=group.group_id,
            title=group.title,
            legs=legs,
            edge_per_set=net_edge,
            cost_per_set=cost_per_set,
            payout_per_set=ONE,
            confidence=_confidence(net_edge, Decimal(str(self.config.basket_min_edge)), total_usd),
            expected_profit_usd=sets * net_edge,
            max_sets=sets,
            metadata={
                "n_outcomes": group.n_outcomes,
                "gross_edge": str(gross_edge),
                "fee_per_set": str(fee_per_set),
                "dislocation": str(dislocation),
                "taker_rate": str(fee.taker_rate),
            },
        )

    def _max_profitable_sets(self, group: MarketGroup, fee: FeeModel, max_depth: Decimal) -> Decimal | None:
        """Find the largest set size whose average cost stays profitable.

        Average cost is non-decreasing in size, so if the full book capacity is
        profitable it is the answer; otherwise the feasible sizes form a prefix
        that can be found by bisection.
        """
        min_edge = Decimal(str(self.config.basket_min_edge))

        def profitable(sets: Decimal) -> bool:
            if sets <= 0:
                return True
            usd = ZERO
            fee_total = ZERO
            for outcome in group.outcomes:
                fill = outcome.book.cost_to_buy(sets, price_limit=ONE)
                if not fill.complete or fill.shares < sets:
                    return False
                usd += fill.usd
                fee_total += fee.buy_fee(sets, fill.avg_price)
            net = ONE - (usd + fee_total) / sets
            return net >= min_edge

        if max_depth <= 0:
            return None
        if profitable(max_depth):
            return max_depth

        low, high = ZERO, max_depth
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
        result = quantize_down(low, ONE)
        return result if result > 0 else None


def _depth_shares(outcome: Outcome) -> Decimal:
    return sum((lvl.size for lvl in outcome.book.asks[:MAX_LEVELS]), ZERO)


@dataclass
class FadeExtremeStrategy:
    """Directional mean reversion on extreme prices.

    Disabled by default. The edge below is a *model assumption*, not an
    arbitrage: we assume the price drifts a fraction of the way back toward
    0.50. Treat every signal from this strategy as a discretionary bet.
    """

    config: Config

    def evaluate(self, group: MarketGroup, fee: FeeModel) -> Signal | None:
        if not group.is_binary or len(group.outcomes) != 2:
            return None
        below = Decimal(str(self.config.fade_price_below))
        max_entry = Decimal(str(self.config.fade_max_entry))
        min_top = Decimal(str(self.config.fade_min_top_size))

        for outcome in group.outcomes:
            book = outcome.book
            if book is None or not book.has_two_sided_market():
                continue
            ask = book.best_ask
            bid = book.best_bid
            if ask is None or bid is None:
                continue
            if ask > below or ask > max_entry:
                continue
            if book.top_ask_size < min_top:
                continue
            # Reversion assumption: price moves `alpha` of the way to 0.50.
            # The resulting "edge" is entirely model-driven, not a market
            # dislocation, so it is reported as an assumption and the
            # confidence is deliberately low.
            alpha = Decimal(str(self.config.fade_reversion_alpha))
            expected_price = ask + alpha * (Decimal("0.5") - ask)
            gross = expected_price - ask
            fee_per_share = fee.buy_fee(ONE, ask)
            net = gross - fee_per_share
            min_edge = Decimal(str(self.config.fade_min_edge))
            if net < min_edge:
                continue
            budget = Decimal(str(self.config.max_order_usd))
            fill = book.shares_for_budget(budget, price_limit=ask)
            shares = quantize_down(fill.shares, ONE)
            if shares <= 0:
                continue
            # Confidence reflects how much of the assumed move is left on the
            # table after costs, capped low because the premise is unverified.
            confidence = round(float(min(Decimal("0.30"), net / Decimal("0.10"))), 4)
            return Signal(
                kind="fade_extreme",
                group_id=group.group_id,
                title=group.title,
                legs=(Leg(outcome.token_id, outcome.name, ask, shares, fill.usd, touch_price=ask),),
                edge_per_set=net,
                cost_per_set=ask,
                # The exit is assumed, so the payout is the assumed exit price,
                # never the $1 face value of a held-to-resolution share.
                payout_per_set=expected_price,
                confidence=confidence,
                expected_profit_usd=shares * net,
                max_sets=shares,
                metadata={
                    "directional": True,
                    "assumed_alpha": str(alpha),
                    "assumed_exit_price": str(expected_price),
                    "taker_rate": str(fee.taker_rate),
                    "warning": "heuristic mean-reversion assumption, not risk-free",
                },
            )
        return None


@dataclass
class StrategyEngine:
    """Run every enabled strategy over a group and apply market filters."""

    config: Config
    set_arbitrage: SetArbitrageStrategy = None  # type: ignore[assignment]
    basket_arbitrage: BasketArbitrageStrategy = None  # type: ignore[assignment]
    fade: FadeExtremeStrategy = None  # type: ignore[assignment]
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.set_arbitrage = SetArbitrageStrategy(self.config)
        self.basket_arbitrage = BasketArbitrageStrategy(self.config)
        self.fade = FadeExtremeStrategy(self.config)

    def evaluate(self, group: MarketGroup, fee: FeeModel) -> list[Signal]:
        signals: list[Signal] = []
        if not self._passes_common_filters(group):
            return signals
        if self.config.arb_enabled:
            signal = self.set_arbitrage.evaluate(group, fee)
            if signal is not None:
                signals.append(signal)
        if self.config.basket_enabled:
            signal = self.basket_arbitrage.evaluate(group, fee)
            if signal is not None:
                signals.append(signal)
        if self.config.fade_enabled:
            signal = self.fade.evaluate(group, fee)
            if signal is not None:
                signals.append(signal)
        return signals

    def _passes_common_filters(self, group: MarketGroup) -> bool:
        seconds = group.seconds_to_end
        if group.is_binary:
            low, high = self.config.arb_min_seconds_left, self.config.arb_max_seconds_left
            min_vol = self.config.arb_min_volume_24h
            min_liq = self.config.arb_min_liquidity
        else:
            low, high = self.config.basket_min_seconds_left, self.config.arb_max_seconds_left
            min_vol = self.config.basket_min_volume_24h
            min_liq = self.config.basket_min_liquidity
        if seconds is not None and (seconds < low or seconds > high):
            return False
        if group.volume_24h < Decimal(str(min_vol)):
            return False
        if group.liquidity < Decimal(str(min_liq)):
            return False
        return True
