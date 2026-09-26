"""Tests for exits: take profit, stop loss, settlement, and the duplicate guard.

These exercise the real code paths (portfolio, risk, exit engine) against
hand-built order books, so a passing test means an actual sell decision was
reached and booked, not that a stub was called.
"""

from __future__ import annotations

from decimal import Decimal

from polymarket_bot.brokers import PaperBroker
from polymarket_bot.config import Config
from polymarket_bot.exits import ExitEngine
from polymarket_bot.fees import FeeModel
from polymarket_bot.models import Leg, MarketGroup, OrderBook, Outcome, Signal
from polymarket_bot.portfolio import Portfolio, Position, PositionLeg

FEE_FREE = FeeModel.for_market(fees_enabled=False)


def make_config(**overrides) -> Config:
    defaults = dict(
        take_profit_pct=0.15,
        stop_loss_pct=0.30,
        exits_enabled=True,
        max_theme_exposure_usd=150.0,
        max_order_usd=100.0,
        max_total_exposure_usd=500.0,
        min_free_balance_usd=0.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def book(bid: str, ask: str, size: str = "1000") -> OrderBook:
    return OrderBook.from_api(
        {
            "bids": [{"price": bid, "size": size}],
            "asks": [{"price": ask, "size": size}],
            "tick_size": "0.01",
        }
    )


def group(yes_bid="0.60", no_bid="0.45", yes_ask="0.62", no_ask="0.47") -> MarketGroup:
    return MarketGroup(
        group_id="g1",
        title="Test market",
        outcomes=(
            Outcome(0, "outcome-0", "tok-0", book(yes_bid, yes_ask)),
            Outcome(1, "outcome-1", "tok-1", book(no_bid, no_ask)),
        ),
        volume_24h=Decimal("100000"),
        liquidity=Decimal("50000"),
        seconds_to_end=86400,
        is_binary=True,
        metadata={"condition_id": "0xcond"},
    )


def arb_signal(sets="100", cost_per_set="0.95", payout="1") -> Signal:
    cost = Decimal(cost_per_set)
    n = Decimal(sets)
    legs = tuple(
        Leg(f"tok-{i}", f"outcome-{i}", cost / 2, n, (cost / 2) * n, touch_price=cost / 2)
        for i in range(2)
    )
    return Signal(
        kind="set_arbitrage",
        group_id="g1",
        title="Test market",
        legs=legs,
        edge_per_set=Decimal(payout) - cost,
        cost_per_set=cost,
        payout_per_set=Decimal(payout),
        confidence=0.8,
        max_sets=n,
    )


def open_set(portfolio: Portfolio, cost="95") -> Position:
    """Open a hedged two-leg position with equal legs."""
    return portfolio.open_position(
        arb_signal(),
        Decimal(cost),
        fills=(
            ("tok-0", "outcome-0", Decimal("100"), Decimal("0.475"), "0xcond"),
            ("tok-1", "outcome-1", Decimal("100"), Decimal("0.475"), "0xcond"),
        ),
        condition_id="0xcond",
    )


def build_exits(**overrides) -> tuple[ExitEngine, Portfolio, PaperBroker]:
    config = make_config(**overrides)
    portfolio = Portfolio(Decimal("1000"))
    broker = PaperBroker(config)
    return ExitEngine(config, broker, portfolio), portfolio, broker


# ------------------------------------------------------------- take profit


def test_take_profit_triggers_when_bids_clear_the_target():
    """Entry 0.95/set, exit 1.05/set: +10.5% net clears a 15% target only if
    the target is 10%. Use an explicit low target so the intent is unambiguous."""
    engine, portfolio, _ = build_exits(take_profit_pct=0.05)
    position = open_set(portfolio)
    decision = engine.evaluate(position, group(yes_bid="0.60", no_bid="0.45"), FEE_FREE)
    assert decision.should_exit
    assert decision.reason == "take_profit"
    # 100 sets sold at 0.60 + 0.45 = 1.05 -> net 105.00 against cost 95.00.
    assert decision.proceeds_usd == Decimal("105.00")
    assert decision.pnl_usd == Decimal("10.00")


def test_take_profit_does_not_trigger_below_target():
    engine, portfolio, _ = build_exits(take_profit_pct=0.15)
    position = open_set(portfolio)
    # 1.05/1.00 exit is +10.5%, under the 15% target.
    decision = engine.evaluate(position, group(yes_bid="0.60", no_bid="0.45"), FEE_FREE)
    assert not decision.should_exit


def test_take_profit_closes_the_position_and_realizes_cash():
    engine, portfolio, broker = build_exits(take_profit_pct=0.05)
    position = open_set(portfolio)
    decision = engine.evaluate(position, group(yes_bid="0.60", no_bid="0.45"), FEE_FREE)
    assert engine.close(position, group(yes_bid="0.60", no_bid="0.45"), decision)

    assert portfolio.open_positions == 0
    assert portfolio.cash == Decimal("1010.00")
    assert portfolio.realized_pnl == Decimal("10.00")
    assert engine.exits_taken == 1
    # Two sell orders actually reached the broker.
    assert len([o for o in broker.orders if o["side"] == "SELL"]) == 2


# --------------------------------------------------------------- stop loss


def test_stop_loss_triggers_when_bids_collapse():
    engine, portfolio, _ = build_exits(stop_loss_pct=0.30)
    position = open_set(portfolio)
    # 0.30 + 0.20 = 0.50/set -> 50.00 against 95.00 cost, about -47%.
    decision = engine.evaluate(position, group(yes_bid="0.30", no_bid="0.20"), FEE_FREE)
    assert decision.should_exit
    assert decision.reason == "stop_loss"
    assert decision.pnl_usd == Decimal("-45.00")


def test_stop_loss_does_not_trigger_inside_the_band():
    engine, portfolio, _ = build_exits(stop_loss_pct=0.30)
    position = open_set(portfolio)
    # 0.90/set is about -5%, well inside a 30% stop.
    decision = engine.evaluate(position, group(yes_bid="0.50", no_bid="0.40"), FEE_FREE)
    assert not decision.should_exit


def test_stop_loss_bounds_the_loss_on_a_directional_position():
    """The whole point of the stop: a fade bet that goes wrong must be cut."""
    engine, portfolio, _ = build_exits(stop_loss_pct=0.30)
    signal = Signal(
        kind="fade_extreme",
        group_id="g1",
        title="Fade bet",
        legs=(Leg("tok-0", "Yes", Decimal("0.02"), Decimal("5000"), Decimal("100"), touch_price=Decimal("0.02")),),
        edge_per_set=Decimal("0.03"),
        cost_per_set=Decimal("0.02"),
        payout_per_set=Decimal("0.05"),
        confidence=0.5,
        max_sets=Decimal("5000"),
        metadata={"directional": True, "theme": "iran"},
    )
    position = portfolio.open_position(
        signal,
        Decimal("100"),
        fills=(("tok-0", "Yes", Decimal("5000"), Decimal("0.02"), "0xcond"),),
        condition_id="0xcond",
    )
    # Bid collapses to 0.005: 25.00 against 100.00 is -75%.
    collapsed = MarketGroup(
        group_id="g1",
        title="Fade bet",
        outcomes=(Outcome(0, "Yes", "tok-0", book("0.005", "0.01", size="10000")),),
        seconds_to_end=86400,
        is_binary=True,
    )
    decision = engine.evaluate(position, collapsed, FEE_FREE)
    assert decision.should_exit
    assert decision.reason == "stop_loss"
    assert decision.pnl_usd == Decimal("-75.00")


# -------------------------------------------------------------- thin books


def test_exit_is_skipped_when_the_book_cannot_absorb_the_size():
    """A quoted profit is not real if nobody is bidding for the size."""
    engine, portfolio, _ = build_exits(take_profit_pct=0.05)
    position = open_set(portfolio)
    thin = MarketGroup(
        group_id="g1",
        title="Test market",
        outcomes=(
            Outcome(0, "outcome-0", "tok-0", book("0.60", "0.62", size="10")),
            Outcome(1, "outcome-1", "tok-1", book("0.45", "0.47", size="10")),
        ),
        seconds_to_end=86400,
        is_binary=True,
    )
    decision = engine.evaluate(position, thin, FEE_FREE)
    assert not decision.should_exit
    assert "thin book" in decision.detail


def test_exit_is_skipped_when_a_leg_has_no_bid():
    engine, portfolio, _ = build_exits(take_profit_pct=0.05)
    position = open_set(portfolio)
    one_sided = MarketGroup(
        group_id="g1",
        title="Test market",
        outcomes=(
            Outcome(0, "outcome-0", "tok-0", book("0.60", "0.62")),
            Outcome(1, "outcome-1", "tok-1", OrderBook.from_api({"asks": [{"price": "0.47", "size": "1000"}]})),
        ),
        seconds_to_end=86400,
        is_binary=True,
    )
    decision = engine.evaluate(position, one_sided, FEE_FREE)
    assert not decision.should_exit


def test_hedged_exit_sells_only_the_guaranteed_set_size():
    """Unequal legs: only min(shares) can be sold as a hedged pair."""
    engine, portfolio, broker = build_exits(take_profit_pct=0.05)
    signal = arb_signal()
    position = portfolio.open_position(
        signal,
        Decimal("95"),
        fills=(
            ("tok-0", "outcome-0", Decimal("100"), Decimal("0.475"), "0xcond"),
            ("tok-1", "outcome-1", Decimal("80"), Decimal("0.475"), "0xcond"),
        ),
        condition_id="0xcond",
    )
    assert position.guaranteed_sets == Decimal("80")
    decision = engine.evaluate(position, group(yes_bid="0.60", no_bid="0.45"), FEE_FREE)
    assert engine.close(position, group(yes_bid="0.60", no_bid="0.45"), decision)
    sells = [o for o in broker.orders if o["side"] == "SELL"]
    assert all(Decimal(o["shares"]) == Decimal("80") for o in sells)


# ------------------------------------------------------------- settlement


def test_settlement_pays_one_dollar_per_winning_share():
    engine, portfolio, _ = build_exits()
    position = open_set(portfolio)
    result = engine.settle(position, ("tok-0",))
    assert result.payout_usd == Decimal("100")
    assert result.pnl_usd == Decimal("5")
    assert result.winning
    assert portfolio.realized_pnl == Decimal("5")
    assert portfolio.open_positions == 0


def test_settlement_of_a_loser_returns_nothing():
    engine, portfolio, _ = build_exits()
    position = open_set(portfolio)
    result = engine.settle(position, ("some-other-token",))
    assert result.payout_usd == Decimal("0")
    assert result.pnl_usd == Decimal("-95")
    assert not result.winning


def test_settlement_of_a_basket_pays_every_winning_leg():
    """A neg-risk basket can have more than one winning outcome."""
    engine, portfolio, _ = build_exits()
    signal = Signal(
        kind="basket_arbitrage",
        group_id="g2",
        title="Basket",
        legs=(
            Leg("b0", "A", Decimal("0.30"), Decimal("100"), Decimal("30"), touch_price=Decimal("0.30")),
            Leg("b1", "B", Decimal("0.30"), Decimal("100"), Decimal("30"), touch_price=Decimal("0.30")),
            Leg("b2", "C", Decimal("0.30"), Decimal("100"), Decimal("30"), touch_price=Decimal("0.30")),
        ),
        edge_per_set=Decimal("0.10"),
        cost_per_set=Decimal("0.90"),
        payout_per_set=Decimal("1"),
        confidence=0.8,
        max_sets=Decimal("100"),
    )
    position = portfolio.open_position(
        signal,
        Decimal("90"),
        fills=(
            ("b0", "A", Decimal("100"), Decimal("0.30"), "c0"),
            ("b1", "B", Decimal("100"), Decimal("0.30"), "c1"),
            ("b2", "C", Decimal("100"), Decimal("0.30"), "c2"),
        ),
    )
    result = engine.settle(position, ("b0", "b2"))
    assert result.payout_usd == Decimal("200")


def test_settlement_matches_legs_by_token_not_by_position():
    """Only the held token should pay, even if a sibling outcome wins."""
    engine, portfolio, _ = build_exits()
    position = open_set(portfolio)
    result = engine.settle(position, ("tok-1",))
    assert result.payout_usd == Decimal("100")


# ---------------------------------------------------------- duplicate guard


def test_portfolio_holds_reports_a_held_market():
    portfolio = Portfolio(Decimal("1000"))
    assert not portfolio.holds("g1")
    open_set(portfolio)
    assert portfolio.holds("g1")
    assert not portfolio.holds("other")


def test_holds_is_ignored_for_an_empty_group_id():
    portfolio = Portfolio(Decimal("1000"))
    assert not portfolio.holds("")


def test_theme_exposure_sums_only_matching_positions():
    portfolio = Portfolio(Decimal("1000"))
    signal = arb_signal()
    portfolio.open_position(signal, Decimal("50"), condition_id="0xc", theme="iran")
    portfolio.open_position(
        Signal(**{**signal.__dict__, "group_id": "g2"}), Decimal("30"), condition_id="0xd", theme="iran"
    )
    portfolio.open_position(
        Signal(**{**signal.__dict__, "group_id": "g3"}), Decimal("20"), condition_id="0xe", theme="sports"
    )
    assert portfolio.theme_exposure("iran") == Decimal("80")
    assert portfolio.theme_exposure("sports") == Decimal("20")
    assert portfolio.theme_exposure("unknown") == Decimal("0")


def test_theme_exposure_drops_when_a_position_closes():
    portfolio = Portfolio(Decimal("1000"))
    position = portfolio.open_position(arb_signal(), Decimal("50"), condition_id="0xc", theme="iran")
    assert portfolio.theme_exposure("iran") == Decimal("50")
    portfolio.close_position(position, Decimal("55"))
    assert portfolio.theme_exposure("iran") == Decimal("0")


# ------------------------------------------------------------ ledger truth


def test_expected_payout_uses_held_shares_not_planned_size():
    """A partial fill must not be valued as if the full order went through."""
    portfolio = Portfolio(Decimal("1000"))
    position = portfolio.open_position(
        arb_signal(),
        Decimal("47.50"),
        fills=(
            ("tok-0", "outcome-0", Decimal("50"), Decimal("0.475"), "0xcond"),
            ("tok-1", "outcome-1", Decimal("50"), Decimal("0.475"), "0xcond"),
        ),
        condition_id="0xcond",
    )
    # 50 complete sets, not the 100 the signal asked for.
    assert position.expected_payout_usd == Decimal("50")


def test_save_serializes_leg_detail(tmp_path):
    portfolio = Portfolio(Decimal("1000"))
    open_set(portfolio)
    target = tmp_path / "state.json"
    portfolio.save(target)
    text = target.read_text()
    assert "entry_price" in text
    assert "condition_id" in text
