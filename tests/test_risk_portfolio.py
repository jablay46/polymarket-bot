"""Tests for risk sizing, exposure limits, and the portfolio ledger."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from polymarket_bot.config import Config
from polymarket_bot.fees import FeeModel
from polymarket_bot.models import Leg, Signal
from polymarket_bot.portfolio import Portfolio
from polymarket_bot.risk import RiskManager

FEE_FREE = FeeModel.for_market(fees_enabled=False)


def make_config(**overrides) -> Config:
    defaults = dict(
        max_order_usd=100.0,
        max_total_exposure_usd=500.0,
        max_orders_per_minute=10,
        min_free_balance_usd=20.0,
        kill_switch=False,
    )
    defaults.update(overrides)
    return Config(**defaults)


def signal(cost_per_set="0.95", sets="100", payout="1", legs=2, kind="set_arbitrage") -> Signal:
    cost = Decimal(cost_per_set)
    n = Decimal(sets)
    legs_tuple = tuple(
        Leg(f"tok-{i}", f"outcome-{i}", cost / legs, n, (cost / legs) * n) for i in range(legs)
    )
    return Signal(
        kind=kind,
        group_id="g1",
        title="t",
        legs=legs_tuple,
        edge_per_set=Decimal(payout) - cost,
        cost_per_set=cost,
        payout_per_set=Decimal(payout),
        confidence=0.8,
        expected_profit_usd=n * (Decimal(payout) - cost),
        max_sets=n,
    )


# ------------------------------------------------------------------ sizing


def test_size_is_capped_by_per_order_limit():
    manager = RiskManager(make_config(max_order_usd=50))
    decision = manager.size_signal(
        signal(), FEE_FREE, available_cash=Decimal("10000"), open_exposure=Decimal("0")
    )
    assert decision.approved
    assert decision.usd <= Decimal("50")
    # 50 / 0.95 = 52 sets, costing 49.40
    assert decision.usd == Decimal("52") * Decimal("0.95")


def test_size_is_capped_by_exposure_headroom():
    manager = RiskManager(make_config(max_order_usd=1000, max_total_exposure_usd=500))
    decision = manager.size_signal(
        signal(), FEE_FREE, available_cash=Decimal("10000"), open_exposure=Decimal("480")
    )
    assert decision.approved
    assert decision.usd <= Decimal("20")


def test_size_is_capped_by_available_cash():
    manager = RiskManager(make_config(max_order_usd=1000, min_free_balance_usd=20))
    decision = manager.size_signal(
        signal(), FEE_FREE, available_cash=Decimal("100"), open_exposure=Decimal("0")
    )
    assert decision.approved
    assert decision.usd <= Decimal("80")


def test_size_rejected_when_exposure_limit_reached():
    manager = RiskManager(make_config(max_total_exposure_usd=500))
    decision = manager.size_signal(
        signal(), FEE_FREE, available_cash=Decimal("10000"), open_exposure=Decimal("500")
    )
    assert not decision.approved
    assert "exposure" in decision.reason


def test_size_rejected_when_balance_below_floor():
    manager = RiskManager(make_config(min_free_balance_usd=20))
    decision = manager.size_signal(
        signal(), FEE_FREE, available_cash=Decimal("15"), open_exposure=Decimal("0")
    )
    assert not decision.approved


def test_kill_switch_blocks_sizing():
    manager = RiskManager(make_config(kill_switch=True))
    decision = manager.size_signal(
        signal(), FEE_FREE, available_cash=Decimal("10000"), open_exposure=Decimal("0")
    )
    assert not decision.approved
    assert "kill switch" in decision.reason


def test_size_rejected_when_budget_cannot_buy_one_set():
    # A $0.50 per-order cap cannot buy even one $0.95 set.
    manager = RiskManager(make_config(max_order_usd=0.50))
    decision = manager.size_signal(
        signal(cost_per_set="0.95"), FEE_FREE, available_cash=Decimal("10000"), open_exposure=Decimal("0")
    )
    assert not decision.approved
    assert "below one set" in decision.reason


def test_fees_reduce_the_size_for_the_same_budget():
    fee_model = FeeModel.for_market(fees_enabled=True, fee_type="crypto_fees_v2")
    manager = RiskManager(make_config(max_order_usd=100))
    # A large max_sets keeps the depth cap from binding, so the budget does.
    free = manager.size_signal(
        signal(sets="1000"), FEE_FREE, available_cash=Decimal("10000"), open_exposure=Decimal("0")
    )
    paid = manager.size_signal(
        signal(sets="1000"), fee_model, available_cash=Decimal("10000"), open_exposure=Decimal("0")
    )
    assert free.approved and paid.approved
    # Same budget, but fees mean fewer whole sets fit.
    assert paid.usd < free.usd


def test_size_rejected_when_open_position_limit_reached():
    manager = RiskManager(make_config(max_open_positions=2))
    decision = manager.size_signal(
        signal(), FEE_FREE, available_cash=Decimal("10000"), open_exposure=Decimal("0"), open_positions=2
    )
    assert not decision.approved
    assert "open positions" in decision.reason


def test_book_impact_cap_shrinks_an_aggressive_size():
    """A size whose average fill is far above the touch must be trimmed."""
    aggressive = signal(cost_per_set="0.95", sets="1000", legs=2)
    # A touch price 1.5x below the average fill is a 50% impact.
    aggressive = replace(
        aggressive,
        legs=tuple(
            Leg(leg.token_id, leg.outcome_name, leg.price, leg.shares, leg.usd, touch_price=leg.price / Decimal("1.5"))
            for leg in aggressive.legs
        ),
    )
    manager = RiskManager(make_config(max_order_usd=10000, max_total_exposure_usd=100000, max_book_impact=0.03))
    decision = manager.size_signal(
        aggressive, FEE_FREE, available_cash=Decimal("100000"), open_exposure=Decimal("0")
    )
    assert decision.approved
    # 50% impact against a 3% cap scales the 1000-set size down to ~60.
    assert decision.usd < Decimal("100")


def test_book_impact_cap_is_inert_for_a_clean_fill():
    clean = signal(cost_per_set="0.95", sets="1000", legs=2)
    clean = replace(
        clean,
        legs=tuple(
            Leg(leg.token_id, leg.outcome_name, leg.price, leg.shares, leg.usd, touch_price=leg.price)
            for leg in clean.legs
        ),
    )
    manager = RiskManager(make_config(max_order_usd=10000, max_total_exposure_usd=100000, max_book_impact=0.03))
    decision = manager.size_signal(
        clean, FEE_FREE, available_cash=Decimal("100000"), open_exposure=Decimal("0")
    )
    assert decision.approved
    assert decision.usd == Decimal("950")


# --------------------------------------------------------------- rate limit


def test_rate_limit_allows_configured_number_then_blocks():
    manager = RiskManager(make_config(max_orders_per_minute=3))
    assert manager.rate_ok()
    assert manager.rate_ok()
    assert manager.rate_ok()
    assert not manager.rate_ok()


def test_can_trade_respects_limits():
    manager = RiskManager(make_config(max_total_exposure_usd=500, min_free_balance_usd=20))
    assert manager.can_trade(Decimal("100"), Decimal("100"))
    assert not manager.can_trade(Decimal("500"), Decimal("100"))
    assert not manager.can_trade(Decimal("100"), Decimal("10"))


# ---------------------------------------------------------------- portfolio


def test_portfolio_starts_with_configured_balance():
    portfolio = Portfolio(Decimal("1000"))
    assert portfolio.cash == Decimal("1000")
    assert portfolio.open_exposure == Decimal("0")
    assert portfolio.equity == Decimal("1000")


def test_opening_a_position_moves_cash_to_exposure():
    portfolio = Portfolio(Decimal("1000"))
    position = portfolio.open_position(signal(), Decimal("95"))
    assert portfolio.cash == Decimal("905")
    assert portfolio.open_exposure == Decimal("95")
    assert portfolio.equity == Decimal("1000")
    assert portfolio.open_positions == 1
    assert position.expected_profit_usd == Decimal("5")


def test_closing_a_position_realizes_profit():
    """A complete set bought for 0.95 pays out 1.00 at resolution."""
    portfolio = Portfolio(Decimal("1000"))
    position = portfolio.open_position(signal(), Decimal("95"))
    # Equity is unchanged while the position is open: cash fell, exposure rose.
    assert portfolio.equity == Decimal("1000")

    pnl = portfolio.close_position(position, Decimal("100"))
    assert pnl == Decimal("5")
    assert portfolio.cash == Decimal("1005")
    assert portfolio.realized_pnl == Decimal("5")
    assert portfolio.open_exposure == Decimal("0")
    assert portfolio.equity == Decimal("1005")
    assert portfolio.open_positions == 0


def test_portfolio_rejects_overspend():
    portfolio = Portfolio(Decimal("10"))
    try:
        portfolio.open_position(signal(), Decimal("95"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on overspend")
    assert portfolio.cash == Decimal("10")


def test_portfolio_snapshot_reports_expected_payout():
    portfolio = Portfolio(Decimal("1000"))
    portfolio.open_position(signal(sets="100", payout="1", cost_per_set="0.95"), Decimal("95"))
    snap = portfolio.snapshot()
    assert Decimal(snap["open_exposure"]) == Decimal("95")
    assert Decimal(snap["hedged_exposure"]) == Decimal("95")
    assert Decimal(snap["directional_exposure"]) == Decimal("0")
    assert Decimal(snap["expected_payout"]) == Decimal("100")
    assert Decimal(snap["expected_profit"]) == Decimal("5")


def test_directional_positions_are_reported_separately():
    """Fade-style positions have an assumed, not certain, payout."""
    portfolio = Portfolio(Decimal("1000"))
    directional = signal(sets="100", payout="0.055", cost_per_set="0.005", kind="fade_extreme")
    directional.metadata["directional"] = True
    position = portfolio.open_position(directional, Decimal("0.50"))

    assert position.hedged is False
    # The assumed profit must not be presented as locked-in profit.
    assert portfolio.expected_profit == Decimal("0")
    assert portfolio.directional_exposure == Decimal("0.50")
    assert portfolio.assumed_profit == Decimal("5")
    assert "assumed_profit" in portfolio.summary()


def test_portfolio_save_writes_json(tmp_path):
    portfolio = Portfolio(Decimal("1000"))
    portfolio.open_position(signal(), Decimal("95"))
    target = tmp_path / "state.json"
    portfolio.save(target)
    assert target.exists()
    assert "open_exposure" in target.read_text()
