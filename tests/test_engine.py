"""Engine cycle tests with a fake scanner (no network)."""

from __future__ import annotations

from decimal import Decimal

from polymarket_bot.brokers import PaperBroker
from polymarket_bot.config import Config
from polymarket_bot.engine import TradingEngine
from polymarket_bot.models import MarketGroup, OrderBook, Outcome
from polymarket_bot.portfolio import Portfolio


def make_config(**overrides) -> Config:
    defaults = dict(
        arb_enabled=True,
        basket_enabled=False,
        fade_enabled=False,
        max_order_usd=100.0,
        max_total_exposure_usd=500.0,
        max_orders_per_minute=10,
        min_free_balance_usd=0.0,
        arb_min_edge=0.02,
        arb_max_edge=0.15,
        arb_max_spread=0.05,
        arb_min_volume_24h=0.0,
        arb_min_liquidity=0.0,
        arb_min_top_size=0.0,
        arb_min_seconds_left=0,
        arb_max_seconds_left=10**9,
        arb_cooldown_seconds=60,
        arb_antichase_spike=10.0,
        paper_starting_balance=1000.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def arb_group(ask_yes="0.45", ask_no="0.50") -> MarketGroup:
    def book(ask, bid):
        return OrderBook.from_api(
            {"asks": [{"price": ask, "size": "500"}], "bids": [{"price": bid, "size": "500"}], "tick_size": "0.01"}
        )

    return MarketGroup(
        group_id="g1",
        title="Arb market",
        outcomes=(
            Outcome(0, "Yes", "yes", book(ask_yes, str(Decimal(ask_yes) - Decimal("0.01")))),
            Outcome(1, "No", "no", book(ask_no, str(Decimal(ask_no) - Decimal("0.01")))),
        ),
        volume_24h=Decimal("100000"),
        liquidity=Decimal("50000"),
        seconds_to_end=86400,
        is_binary=True,
    )


class FakeScanner:
    def __init__(self, groups):
        self._groups = groups
        self._fee_cache: dict = {}

    def scan(self):
        return self._groups


class ShiftingScanner(FakeScanner):
    """Returns a group whose prices move between scans."""

    def __init__(self, groups):
        super().__init__(groups)
        self.calls = 0

    def scan(self):
        self.calls += 1
        return [arb_group(ask_yes=f"0.45{self.calls - 1}", ask_no="0.50")]


def build_engine(groups, **overrides):
    config = make_config(**overrides)
    engine = TradingEngine(config)
    engine.scanner = FakeScanner(groups)
    engine.broker = PaperBroker(config)
    from polymarket_bot.execution import ExecutionEngine

    engine.portfolio = Portfolio(Decimal("1000"))
    engine.execution = ExecutionEngine(config, engine.broker, engine.portfolio)
    return engine


def test_cycle_executes_a_profitable_opportunity():
    engine = build_engine([arb_group()])
    stats = engine.run_cycle()
    assert stats.signals_found >= 1
    assert stats.orders_filled == 1
    assert stats.notional_usd > 0
    assert engine.portfolio.open_positions == 1


def test_cycle_skips_opportunity_inside_cooldown():
    engine = build_engine([arb_group()])
    engine.run_cycle()
    second = engine.run_cycle()
    assert second.orders_filled == 0


def test_cycle_respects_kill_switch():
    engine = build_engine([arb_group()], kill_switch=True)
    stats = engine.run_cycle()
    assert stats.signals_found >= 1
    assert stats.orders_filled == 0
    assert stats.signals_rejected >= 1


def test_cycle_finds_nothing_when_no_arbitrage_exists():
    engine = build_engine([arb_group(ask_yes="0.55", ask_no="0.50")])
    stats = engine.run_cycle()
    assert stats.orders_filled == 0


def test_cycle_survives_a_scanner_exception():
    class BrokenScanner:
        def scan(self):
            raise RuntimeError("network down")

    engine = build_engine([])
    engine.scanner = BrokenScanner()
    stats = engine.run_cycle()
    assert stats.errors == 1
    assert stats.orders_filled == 0


def test_anti_chase_blocks_a_price_spike():
    """A price that jumps between cycles must not be chased."""
    config = make_config(arb_antichase_spike=0.001)
    engine = TradingEngine(config)
    engine.scanner = ShiftingScanner([])
    engine.broker = PaperBroker(config)
    engine.portfolio = Portfolio(Decimal("1000"))
    from polymarket_bot.execution import ExecutionEngine

    engine.execution = ExecutionEngine(config, engine.broker, engine.portfolio)

    first = engine.run_cycle()
    assert first.orders_filled == 1
    # Clear the cooldown so only the anti-chase guard can block the re-entry.
    engine._cooldowns.clear()
    second = engine.run_cycle()
    assert second.signals_found >= 1
    assert second.orders_filled == 0
    assert second.signals_rejected >= 1


def test_anti_chase_allows_a_stable_price():
    config = make_config(arb_antichase_spike=0.5)
    engine = TradingEngine(config)
    engine.scanner = FakeScanner([arb_group()])
    engine.broker = PaperBroker(config)
    engine.portfolio = Portfolio(Decimal("1000"))
    from polymarket_bot.execution import ExecutionEngine

    engine.execution = ExecutionEngine(config, engine.broker, engine.portfolio)
    engine.run_cycle()
    engine._cooldowns.clear()
    # Close the position so the duplicate guard is not what decides this;
    # a stable price must pass the anti-chase check and re-enter cleanly.
    for position in list(engine.portfolio.positions):
        engine.portfolio.close_position(position, position.cost_usd)
    second = engine.run_cycle()
    assert second.orders_filled == 1


def test_cycle_never_doubles_up_on_a_market_it_already_holds():
    """The cooldown is a timer; holding the market must block re-entry forever."""
    engine = build_engine([arb_group()])
    first = engine.run_cycle()
    assert first.orders_filled == 1

    # Clear the cooldown so only the held-position guard can prevent a re-buy.
    engine._cooldowns.clear()
    second = engine.run_cycle()
    assert second.signals_found >= 1
    assert second.orders_filled == 0
    assert second.signals_rejected >= 1
    assert engine.portfolio.open_positions == 1


def test_cycle_reenters_after_the_position_is_closed():
    """Closing a position must free the market for a fresh entry."""
    engine = build_engine([arb_group()])
    engine.run_cycle()
    position = engine.portfolio.positions[0]
    engine.portfolio.close_position(position, position.cost_usd)
    engine._cooldowns.clear()
    stats = engine.run_cycle()
    assert stats.orders_filled == 1
    assert engine.portfolio.open_positions == 1


def test_run_forever_stops_after_max_cycles():
    engine = build_engine([arb_group()], poll_interval_seconds=1)
    engine.run_forever(max_cycles=2)
    # Two cycles ran, so exactly one order was placed (the second was cooled down).
    assert engine.portfolio.open_positions == 1


# ----------------------------------------------------- cross-market wiring


def ladder_group(threshold: str, condition_id: str, tick: str = "0.01") -> MarketGroup:
    """A binary market whose question carries a BTC threshold."""
    def book(ask, bid):
        return OrderBook.from_api(
            {"asks": [{"price": ask, "size": "500"}], "bids": [{"price": bid, "size": "500"}], "tick_size": tick}
        )

    return MarketGroup(
        group_id=condition_id,
        title=f"Will Bitcoin be above ${threshold} by December 31?",
        outcomes=(
            Outcome(0, "Yes", f"yes-{threshold}", book("0.45", "0.44")),
            Outcome(1, "No", f"no-{threshold}", book("0.45", "0.44")),
        ),
        volume_24h=Decimal("100000"),
        liquidity=Decimal("50000"),
        seconds_to_end=86400,
        is_binary=True,
        tick_size=Decimal(tick),
        metadata={"condition_id": condition_id, "market_id": condition_id},
    )


class LadderScanner(FakeScanner):
    """Serves books for ladder tokens so the cross-market path can price."""

    def fetch_books(self, token_ids):
        books = {}
        for group in self._groups:
            for outcome in group.outcomes:
                if outcome.token_id in token_ids:
                    books[outcome.token_id] = outcome.book
        return books


def build_ladder_engine(groups, **overrides):
    # Only the cross-market strategy is under test here; the per-market
    # strategies are switched off so a set-arb on the same books cannot
    # account for the fills being asserted on.
    defaults = dict(
        cross_market_enabled=True,
        arb_enabled=False,
        basket_enabled=False,
        fade_enabled=False,
    )
    defaults.update(overrides)
    config = make_config(**defaults)
    engine = TradingEngine(config)
    engine.scanner = LadderScanner(groups)
    engine.broker = PaperBroker(config)
    from polymarket_bot.execution import ExecutionEngine

    engine.portfolio = Portfolio(Decimal("1000"))
    engine.execution = ExecutionEngine(config, engine.broker, engine.portfolio)
    return engine


def test_cross_market_cycle_trades_a_priced_violation_in_paper_mode():
    groups = [ladder_group("90,000", "0xlo"), ladder_group("100,000", "0xhi")]
    engine = build_ladder_engine(groups)
    stats = engine.run_cycle()
    assert stats.orders_filled == 1
    assert engine.portfolio.open_positions == 1


def test_cross_market_is_off_by_default():
    groups = [ladder_group("90,000", "0xlo"), ladder_group("100,000", "0xhi")]
    engine = build_ladder_engine(groups, cross_market_enabled=False)
    stats = engine.run_cycle()
    assert stats.orders_filled == 0


def test_cross_market_cycle_refuses_unconfirmed_pairs_in_live_mode():
    """The end-to-end guard: even with everything else enabled, live mode must
    not trade a ladder whose relation no human has confirmed.

    The engine runs in paper mode here (a live broker needs credentials), and
    only the cross-market strategy is switched to a live-mode config, which is
    exactly the gate under test.
    """
    from polymarket_bot.cross_market import CrossMarketArbitrageStrategy

    groups = [ladder_group("90,000", "0xlo"), ladder_group("100,000", "0xhi")]
    engine = build_ladder_engine(groups)
    engine.cross_market = CrossMarketArbitrageStrategy(make_config(mode="live"))
    stats = engine.run_cycle()
    assert stats.orders_filled == 0


def test_cross_market_cycle_trades_a_confirmed_pair_in_live_mode():
    from polymarket_bot.cross_market import CrossMarketArbitrageStrategy

    groups = [ladder_group("90,000", "0xlo"), ladder_group("100,000", "0xhi")]
    engine = build_ladder_engine(groups)
    engine.cross_market = CrossMarketArbitrageStrategy(
        make_config(mode="live", cross_market_confirmed_pairs="0xlo:0xhi")
    )
    stats = engine.run_cycle()
    assert stats.orders_filled == 1


def test_cross_market_position_records_both_condition_ids():
    """One position spans two markets, so settlement has to know both."""
    groups = [ladder_group("90,000", "0xlo"), ladder_group("100,000", "0xhi")]
    engine = build_ladder_engine(groups)
    engine.run_cycle()
    position = engine.portfolio.positions[0]
    assert {leg.condition_id for leg in position.legs} == {"0xlo", "0xhi"}
    assert set(engine._condition_ids(position)) == {"0xlo", "0xhi"}


def test_cross_market_does_not_rebuy_a_held_pair():
    """The duplicate guard must cover cross-market positions, or a re-scan
    would stack a second set on top of the first."""
    groups = [ladder_group("90,000", "0xlo"), ladder_group("100,000", "0xhi")]
    engine = build_ladder_engine(groups)
    engine.run_cycle()
    engine._cooldowns.clear()
    second = engine.run_cycle()
    assert second.orders_filled == 0
    assert engine.portfolio.open_positions == 1
