"""Tests for the end-to-end execution path using the paper broker.

These exercise real code paths: the paper broker, the execution engine's
leg-risk handling, and the engine's cycle loop, with only the network layer
replaced by an in-process fake scanner.
"""

from __future__ import annotations

from decimal import Decimal

from polymarket_bot.brokers import LegResult, PaperBroker
from polymarket_bot.config import Config
from polymarket_bot.execution import ExecutionEngine
from polymarket_bot.fees import FeeModel
from polymarket_bot.models import MarketGroup, OrderBook, Outcome
from polymarket_bot.portfolio import Portfolio

FEE_FREE = FeeModel.for_market(fees_enabled=False)


def make_config(**overrides) -> Config:
    defaults = dict(
        max_order_usd=100.0,
        max_total_exposure_usd=500.0,
        paper_starting_balance=1000.0,
        live_order_type="FAK",
    )
    defaults.update(overrides)
    return Config(**defaults)


def book(asks, bids) -> OrderBook:
    return OrderBook.from_api(
        {
            "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
            "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
            "tick_size": "0.01",
        }
    )


def binary_group() -> MarketGroup:
    return MarketGroup(
        group_id="g1",
        title="Test",
        outcomes=(
            Outcome(0, "Yes", "yes", book([("0.45", "500")], [("0.44", "500")])),
            Outcome(1, "No", "no", book([("0.50", "500")], [("0.49", "500")])),
        ),
        is_binary=True,
    )


def make_signal(group: MarketGroup, sets="100"):
    from polymarket_bot.models import Leg, Signal

    n = Decimal(sets)
    return Signal(
        kind="set_arbitrage",
        group_id=group.group_id,
        title=group.title,
        legs=(
            Leg("yes", "Yes", Decimal("0.45"), n, Decimal("0.45") * n),
            Leg("no", "No", Decimal("0.50"), n, Decimal("0.50") * n),
        ),
        edge_per_set=Decimal("0.05"),
        cost_per_set=Decimal("0.95"),
        payout_per_set=Decimal("1"),
        confidence=0.9,
        expected_profit_usd=n * Decimal("0.05"),
        max_sets=n,
    )


def test_paper_broker_fills_and_reports_notional():
    broker = PaperBroker(make_config())
    result = broker.buy(token_id="yes", usd=Decimal("45"), price=Decimal("0.45"))
    assert result.ok
    assert result.filled_usd == Decimal("45")
    assert result.filled_shares == Decimal("100")


def test_execution_places_all_legs_and_records_position():
    portfolio = Portfolio(Decimal("1000"))
    engine = ExecutionEngine(make_config(), PaperBroker(make_config()), portfolio)
    group = binary_group()
    result = engine.execute(make_signal(group), Decimal("95"), group, FEE_FREE)
    assert result.ok
    assert len(result.legs) == 2
    assert all(leg.ok for leg in result.legs)
    assert portfolio.open_positions == 1
    assert portfolio.cash < Decimal("1000")


def test_execution_rejects_non_positive_notional():
    engine = ExecutionEngine(make_config(), PaperBroker(make_config()), Portfolio(Decimal("1000")))
    result = engine.execute(make_signal(binary_group()), Decimal("0"), binary_group(), FEE_FREE)
    assert not result.ok
    assert "notional" in result.error


def test_leg_risk_triggers_unwind_when_a_leg_fails():
    """A failing later leg must trigger a sell of the leg already filled.

    Execution submits the largest-notional leg first, so with notional 50 (No)
    and 45 (Yes) the No leg fills first and the Yes leg is the one that fails.
    """

    class FailingBroker(PaperBroker):
        def __init__(self, config, fail_token: str):
            super().__init__(config)
            self.fail_token = fail_token
            self.buys: list[str] = []
            self.sells: list[str] = []

        def buy(self, *, token_id, usd, price, tick_size=Decimal("0.01"), neg_risk=False, order_type="FAK"):
            self.buys.append(token_id)
            if token_id == self.fail_token:
                return LegResult(token_id, "BUY", False, error="simulated rejection")
            return super().buy(
                token_id=token_id, usd=usd, price=price, tick_size=tick_size, neg_risk=neg_risk, order_type=order_type
            )

        def sell(self, *, token_id, shares, price, tick_size=Decimal("0.01"), neg_risk=False, order_type="FAK"):
            self.sells.append(token_id)
            return super().sell(
                token_id=token_id, shares=shares, price=price, tick_size=tick_size, neg_risk=neg_risk, order_type=order_type
            )

    config = make_config()
    broker = FailingBroker(config, fail_token="yes")
    engine = ExecutionEngine(config, broker, Portfolio(Decimal("1000")))
    group = binary_group()
    result = engine.execute(make_signal(group), Decimal("95"), group, FEE_FREE)

    assert not result.ok
    # The larger No leg went first and filled; the smaller Yes leg failed.
    assert broker.buys == ["no", "yes"]
    assert result.unwound is True
    assert broker.sells == ["no"]
    assert engine.leg_risk_events == 0


def test_leg_risk_is_flagged_when_unwind_also_fails():
    class BrokenBroker(PaperBroker):
        def buy(self, *, token_id, usd, price, tick_size=Decimal("0.01"), neg_risk=False, order_type="FAK"):
            if token_id == "yes":
                return LegResult(token_id, "BUY", False, error="rejected")
            return super().buy(
                token_id=token_id, usd=usd, price=price, tick_size=tick_size, neg_risk=neg_risk, order_type=order_type
            )

        def sell(self, *, token_id, shares, price, tick_size=Decimal("0.01"), neg_risk=False, order_type="FAK"):
            return LegResult(token_id, "SELL", False, error="unwind rejected")

    config = make_config()
    engine = ExecutionEngine(config, BrokenBroker(config), Portfolio(Decimal("1000")))
    result = engine.execute(make_signal(binary_group()), Decimal("95"), binary_group(), FEE_FREE)
    assert not result.ok
    assert result.unwound is False
    assert engine.leg_risk_events == 1


def test_execution_scales_down_when_notional_is_smaller_than_signal():
    portfolio = Portfolio(Decimal("1000"))
    engine = ExecutionEngine(make_config(), PaperBroker(make_config()), portfolio)
    group = binary_group()
    result = engine.execute(make_signal(group, sets="100"), Decimal("9.5"), group, FEE_FREE)
    assert result.ok
    # 10% of the original 100-set signal.
    assert result.notional_usd < Decimal("10")
    assert portfolio.open_positions == 1
