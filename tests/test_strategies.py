"""Tests for the arbitrage and fade strategies."""

from __future__ import annotations

from decimal import Decimal

import pytest

from polymarket_bot.config import Config
from polymarket_bot.fees import FeeModel
from polymarket_bot.models import MarketGroup, OrderBook, Outcome, ZERO
from polymarket_bot.strategies import (
    BasketArbitrageStrategy,
    FadeExtremeStrategy,
    SetArbitrageStrategy,
    StrategyEngine,
)


def make_config(**overrides) -> Config:
    defaults = dict(
        arb_min_edge=0.02,
        arb_max_edge=0.15,
        arb_max_spread=0.05,
        arb_min_volume_24h=0.0,
        arb_min_liquidity=0.0,
        arb_min_top_size=0.0,
        arb_min_seconds_left=0,
        arb_max_seconds_left=10**9,
        basket_min_edge=0.015,
        basket_max_edge=0.25,
        basket_min_outcomes=3,
        basket_max_outcomes=12,
        basket_min_volume_24h=0.0,
        basket_min_liquidity=0.0,
        basket_min_top_size=0.0,
        basket_min_seconds_left=0,
        fade_enabled=True,
        fade_min_volume_24h=0.0,
        fade_min_seconds_left=0,
        fade_max_entry=0.06,
        fade_min_top_size=0.0,
        fade_reversion_alpha=0.10,
        fade_min_edge=0.01,
        max_order_usd=100.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def book(asks, bids, tick="0.01", min_size="5", neg_risk=False) -> OrderBook:
    return OrderBook.from_api(
        {
            "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
            "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
            "tick_size": tick,
            "min_order_size": min_size,
            "neg_risk": neg_risk,
        }
    )


def binary_group(yes_asks, yes_bids, no_asks, no_bids, **kwargs) -> MarketGroup:
    yes = Outcome(0, "Yes", "yes-token", book(yes_asks, yes_bids))
    no = Outcome(1, "No", "no-token", book(no_asks, no_bids))
    params = dict(
        group_id="g1",
        title="Test binary market",
        outcomes=(yes, no),
        volume_24h=Decimal("50000"),
        liquidity=Decimal("20000"),
        seconds_to_end=3600,
        is_binary=True,
    )
    params.update(kwargs)
    return MarketGroup(**params)


FEE_FREE = FeeModel.for_market(fees_enabled=False)
FEE_5 = FeeModel.for_market(fees_enabled=True, fee_type="other_fees")


# --------------------------------------------------------------- set arb


def test_set_arbitrage_detects_clear_opportunity():
    group = binary_group(
        yes_asks=[("0.45", "500")], yes_bids=[("0.44", "500")],
        no_asks=[("0.50", "500")], no_bids=[("0.49", "500")],
    )
    signal = SetArbitrageStrategy(make_config()).evaluate(group, FEE_FREE)
    assert signal is not None
    assert signal.kind == "set_arbitrage"
    assert signal.edge_per_set == pytest.approx(Decimal("0.05"), abs=Decimal("0.0001"))
    assert signal.max_sets == Decimal("500")
    assert len(signal.legs) == 2


def test_set_arbitrage_rejects_when_total_exceeds_one():
    group = binary_group(
        yes_asks=[("0.55", "500")], yes_bids=[("0.54", "500")],
        no_asks=[("0.55", "500")], no_bids=[("0.54", "500")],
    )
    assert SetArbitrageStrategy(make_config()).evaluate(group, FEE_FREE) is None


def test_set_arbitrage_rejects_edge_below_minimum():
    # Cost 0.99 leaves a 1% edge, under the 2% default.
    group = binary_group(
        yes_asks=[("0.49", "500")], yes_bids=[("0.485", "500")],
        no_asks=[("0.50", "500")], no_bids=[("0.495", "500")],
    )
    assert SetArbitrageStrategy(make_config()).evaluate(group, FEE_FREE) is None


def test_set_arbitrage_rejects_absurdly_good_book():
    # A 40% edge is a broken book, above the 15% sanity ceiling.
    group = binary_group(
        yes_asks=[("0.30", "500")], yes_bids=[("0.29", "500")],
        no_asks=[("0.30", "500")], no_bids=[("0.29", "500")],
    )
    assert SetArbitrageStrategy(make_config()).evaluate(group, FEE_FREE) is None


def test_set_arbitrage_rejects_wide_spread():
    group = binary_group(
        yes_asks=[("0.45", "500")], yes_bids=[("0.30", "500")],
        no_asks=[("0.50", "500")], no_bids=[("0.49", "500")],
    )
    assert SetArbitrageStrategy(make_config()).evaluate(group, FEE_FREE) is None


def test_set_arbitrage_rejects_thin_top_of_book():
    group = binary_group(
        yes_asks=[("0.45", "10")], yes_bids=[("0.44", "500")],
        no_asks=[("0.50", "500")], no_bids=[("0.49", "500")],
    )
    assert SetArbitrageStrategy(make_config(arb_min_top_size=100)).evaluate(group, FEE_FREE) is None


def test_set_arbitrage_sizing_is_bounded_by_expensive_deeper_levels():
    # Only the first 50 Yes shares are cheap; the rest sit at 0.95, which makes
    # a complete set cost more than a dollar. Sizing must stay near the cheap
    # band, not run out to the full 1000-share book depth.
    group = binary_group(
        yes_asks=[("0.45", "50"), ("0.95", "1000")], yes_bids=[("0.44", "1000")],
        no_asks=[("0.50", "1000")], no_bids=[("0.49", "1000")],
    )
    signal = SetArbitrageStrategy(make_config()).evaluate(group, FEE_FREE)
    assert signal is not None
    assert signal.max_sets <= Decimal("60")
    assert signal.max_sets >= Decimal("50")
    # Every reported size must still clear the minimum edge on average.
    assert signal.edge_per_set >= Decimal("0.02")


def test_set_arbitrage_fees_reduce_the_edge():
    group = binary_group(
        yes_asks=[("0.45", "500")], yes_bids=[("0.44", "500")],
        no_asks=[("0.50", "500")], no_bids=[("0.49", "500")],
    )
    gross = SetArbitrageStrategy(make_config()).evaluate(group, FEE_FREE)
    net = SetArbitrageStrategy(make_config()).evaluate(group, FEE_5)
    assert net is not None and gross is not None
    assert net.edge_per_set < gross.edge_per_set


def test_set_arbitrage_needs_two_sided_books():
    group = binary_group(
        yes_asks=[("0.45", "500")], yes_bids=[],
        no_asks=[("0.50", "500")], no_bids=[("0.49", "500")],
    )
    assert SetArbitrageStrategy(make_config()).evaluate(group, FEE_FREE) is None


# ------------------------------------------------------------ basket arb


def basket_group(asks, size="500", neg_risk=True) -> MarketGroup:
    outcomes = []
    for i, (name, price) in enumerate(asks):
        outcomes.append(
            Outcome(i, name, f"tok-{i}", book([(price, size)], [(Decimal(price) - Decimal("0.005"), size)], tick="0.001", neg_risk=neg_risk))
        )
    return MarketGroup(
        group_id="basket-1",
        title="Who wins the league?",
        outcomes=tuple(outcomes),
        volume_24h=Decimal("100000"),
        liquidity=Decimal("50000"),
        seconds_to_end=86400,
        neg_risk=True,
        is_binary=False,
    )


def test_basket_arbitrage_detects_undercosted_outcome_set():
    # Six outcomes summing to 0.96 -> 4% edge.
    group = basket_group([("A", "0.30"), ("B", "0.25"), ("C", "0.15"), ("D", "0.12"), ("E", "0.08"), ("F", "0.06")])
    signal = BasketArbitrageStrategy(make_config()).evaluate(group, FEE_FREE)
    assert signal is not None
    assert signal.kind == "basket_arbitrage"
    assert signal.edge_per_set == pytest.approx(Decimal("0.04"), abs=Decimal("0.0005"))
    assert len(signal.legs) == 6


def test_basket_arbitrage_rejects_when_sum_meets_or_exceeds_one():
    group = basket_group([("A", "0.40"), ("B", "0.35"), ("C", "0.30")])
    assert BasketArbitrageStrategy(make_config()).evaluate(group, FEE_FREE) is None


def test_basket_arbitrage_rejects_large_dislocation():
    # Sum 0.70 is a 30% dislocation: treat it as a data problem, not free money.
    group = basket_group([("A", "0.30"), ("B", "0.20"), ("C", "0.20")])
    assert BasketArbitrageStrategy(make_config(basket_max_dislocation=0.15)).evaluate(group, FEE_FREE) is None


def test_basket_arbitrage_requires_enough_outcomes():
    group = basket_group([("A", "0.45"), ("B", "0.45")])
    assert BasketArbitrageStrategy(make_config(basket_min_outcomes=3)).evaluate(group, FEE_FREE) is None


def test_basket_arbitrage_sizing_respects_smallest_leg():
    outcomes = [
        Outcome(0, "A", "t0", book([("0.30", "500")], [("0.295", "500")], tick="0.001")),
        Outcome(1, "B", "t1", book([("0.25", "500")], [("0.245", "500")], tick="0.001")),
        Outcome(2, "C", "t2", book([("0.30", "20")], [("0.295", "500")], tick="0.001")),
    ]
    group = MarketGroup(
        group_id="b2", title="t", outcomes=tuple(outcomes),
        volume_24h=Decimal("100000"), liquidity=Decimal("50000"),
        seconds_to_end=86400, neg_risk=True, is_binary=False,
    )
    signal = BasketArbitrageStrategy(make_config()).evaluate(group, FEE_FREE)
    assert signal is not None
    assert signal.max_sets == Decimal("20")


# ------------------------------------------------------------ fade extreme


def test_fade_finds_cheap_side_when_enabled():
    group = binary_group(
        yes_asks=[("0.02", "5000")], yes_bids=[("0.015", "5000")],
        no_asks=[("0.98", "5000")], no_bids=[("0.975", "5000")],
    )
    signal = FadeExtremeStrategy(make_config()).evaluate(group, FEE_FREE)
    assert signal is not None
    assert signal.kind == "fade_extreme"
    assert signal.metadata["directional"] is True
    assert len(signal.legs) == 1
    # The payout is the assumed exit price, not the $1 face value, so the
    # reported edge cannot exceed the assumed reversion.
    assert signal.payout_per_set < Decimal("0.10")
    assert signal.edge_per_set == signal.payout_per_set - signal.cost_per_set
    assert signal.confidence <= 0.30


def test_fade_ignores_mid_priced_markets():
    group = binary_group(
        yes_asks=[("0.40", "5000")], yes_bids=[("0.39", "5000")],
        no_asks=[("0.60", "5000")], no_bids=[("0.59", "5000")],
    )
    assert FadeExtremeStrategy(make_config()).evaluate(group, FEE_FREE) is None


def test_fade_disabled_by_default_in_engine():
    group = binary_group(
        yes_asks=[("0.02", "5000")], yes_bids=[("0.015", "5000")],
        no_asks=[("0.98", "5000")], no_bids=[("0.975", "5000")],
    )
    engine = StrategyEngine(make_config(fade_enabled=False, arb_enabled=False, basket_enabled=False))
    assert engine.evaluate(group, FEE_FREE) == []


# ---------------------------------------------------------------- engine


def test_engine_applies_volume_filter():
    group = binary_group(
        yes_asks=[("0.45", "500")], yes_bids=[("0.44", "500")],
        no_asks=[("0.50", "500")], no_bids=[("0.49", "500")],
        volume_24h=Decimal("100"),
    )
    engine = StrategyEngine(make_config(arb_min_volume_24h=10000))
    assert engine.evaluate(group, FEE_FREE) == []


def test_engine_applies_time_to_expiry_filter():
    group = binary_group(
        yes_asks=[("0.45", "500")], yes_bids=[("0.44", "500")],
        no_asks=[("0.50", "500")], no_bids=[("0.49", "500")],
        seconds_to_end=60,
    )
    engine = StrategyEngine(make_config(arb_min_seconds_left=600))
    assert engine.evaluate(group, FEE_FREE) == []


def test_engine_returns_both_arbitrage_signals_when_applicable():
    # A binary market is also a two-outcome basket, but basket arbitrage
    # requires >= 3 outcomes, so only the set-arbitrage signal should appear.
    group = binary_group(
        yes_asks=[("0.45", "500")], yes_bids=[("0.44", "500")],
        no_asks=[("0.50", "500")], no_bids=[("0.49", "500")],
    )
    engine = StrategyEngine(make_config())
    signals = engine.evaluate(group, FEE_FREE)
    assert [s.kind for s in signals] == ["set_arbitrage"]
