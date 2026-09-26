"""The main trading loop: scan, evaluate, size, execute, report.

Each cycle is independent and fault-isolated. A failing market, a failing
strategy, or a failing order logs and moves on rather than taking the loop
down. Cooldowns and an anti-chase guard keep the bot from re-entering the same
market repeatedly or chasing a price that just spiked.
"""

from __future__ import annotations

import signal as signal_module
import time
from dataclasses import dataclass, field
from decimal import Decimal

from .brokers import BrokerError, build_broker
from .config import Config
from .data import MarketScanner
from .execution import ExecutionEngine, ExecutionResult
from .exits import ExitEngine
from .fees import FeeModel
from .logging_setup import get_logger
from .models import ONE, ZERO, MarketGroup, Outcome, Signal
from .portfolio import Portfolio
from .risk import RiskManager
from .strategies import StrategyEngine

log = get_logger("engine")


@dataclass
class CycleStats:
    markets_scanned: int = 0
    groups_evaluated: int = 0
    signals_found: int = 0
    signals_rejected: int = 0
    orders_filled: int = 0
    errors: int = 0
    best_edge: Decimal = ZERO
    notional_usd: Decimal = ZERO
    exits: int = 0
    stop_losses: int = 0
    settled: int = 0
    realized_usd: Decimal = ZERO

    def as_dict(self) -> dict:
        return {
            "markets": self.markets_scanned,
            "groups": self.groups_evaluated,
            "signals": self.signals_found,
            "rejected": self.signals_rejected,
            "filled": self.orders_filled,
            "errors": self.errors,
            "best_edge": float(self.best_edge),
            "notional": float(self.notional_usd),
            "exits": self.exits,
            "stopped": self.stop_losses,
            "settled": self.settled,
        }


@dataclass
class TradingEngine:
    """Wire the scanner, strategies, risk manager, and execution together."""

    config: Config
    scanner: MarketScanner = None  # type: ignore[assignment]
    strategies: StrategyEngine = None  # type: ignore[assignment]
    risk: RiskManager = None  # type: ignore[assignment]
    portfolio: Portfolio = None  # type: ignore[assignment]
    execution: ExecutionEngine = None  # type: ignore[assignment]
    broker: object = None
    exits: ExitEngine = None  # type: ignore[assignment]
    _cooldowns: dict = field(default_factory=dict)
    _price_memory: dict = field(default_factory=dict)
    _themes: dict = field(default_factory=dict)
    _stopping: bool = False

    def __post_init__(self) -> None:
        if self.scanner is None:
            self.scanner = MarketScanner(self.config)
        if self.strategies is None:
            self.strategies = StrategyEngine(self.config)
        if self.risk is None:
            self.risk = RiskManager(self.config)
        if self.portfolio is None:
            self.portfolio = Portfolio(Decimal(str(self.config.paper_starting_balance)))
        if self.broker is None:
            self.broker = build_broker(self.config)
        if self.execution is None:
            self.execution = ExecutionEngine(self.config, self.broker, self.portfolio)
        if self.exits is None:
            self.exits = ExitEngine(self.config, self.broker, self.portfolio, self.scanner)

    # ---------------------------------------------------------------- helpers
    def _fee_model(self, group: MarketGroup) -> FeeModel:
        override = self.config.taker_fee_rate_override or None
        if override is None and group.is_binary and group.metadata.get("market_id"):
            # Reuse the scanner's cached category lookup when available.
            rate = getattr(self.scanner, "_fee_cache", {}).get(group.metadata.get("fee_type") or "default")
            if rate is not None:
                override = rate
        return FeeModel.for_market(
            fees_enabled=group.fees_enabled,
            fee_type=group.metadata.get("fee_type"),
            override=override,
            safety_multiplier=self.config.fee_safety_multiplier,
            maker=self.config.assume_maker,
        )

    def _on_cooldown(self, group_id: str, now: float) -> bool:
        until = self._cooldowns.get(group_id, 0.0)
        return now < until

    def _set_cooldown(self, group_id: str, now: float) -> None:
        self._cooldowns[group_id] = now + self.config.arb_cooldown_seconds

    def _theme_for(self, group: MarketGroup) -> str:
        """Correlation theme for a group, resolved from its event's rarest tag.

        Cached per event id so a theme lookup costs one request per event, not
        one per cycle.
        """
        if not self.config.max_theme_exposure_usd:
            return ""
        event_id = str(group.metadata.get("event_id") or "")
        if not event_id:
            return ""
        if event_id in self._themes:
            return self._themes[event_id]
        try:
            found = self.scanner.fetch_themes([event_id])
        except Exception as exc:  # noqa: BLE001 - theming must never block trading
            log.debug("theme lookup failed for %s: %s", event_id, exc)
            found = {}
        theme = found.get(event_id, "")
        self._themes[event_id] = theme
        return theme

    def _chased(self, group_id: str, signal: Signal) -> bool:
        """Reject a signal whose price jumped since the previous cycle.

        A sudden move usually means we are looking at a stale or fading quote
        rather than a genuine dislocation.
        """
        key = f"{group_id}:{signal.kind}"
        previous = self._price_memory.get(key)
        current = signal.cost_per_set
        self._price_memory[key] = current
        if previous is None or previous <= 0:
            return False
        spike = abs(current - previous) / previous
        return spike > Decimal(str(self.config.arb_antichase_spike))

    def _open_exposure(self) -> Decimal:
        if self.portfolio is not None:
            return self.portfolio.open_exposure
        return ZERO

    def _available_cash(self) -> Decimal:
        if self.config.is_live:
            return self.broker.balance_usd()
        if self.portfolio is not None:
            return self.portfolio.cash
        return ZERO

    # ------------------------------------------------------------------ cycle
    def run_cycle(self) -> CycleStats:
        stats = CycleStats()
        now = time.monotonic()

        try:
            groups = self.scanner.scan()
        except Exception as exc:  # noqa: BLE001 - the loop must survive
            log.error("scan failed: %s", exc)
            stats.errors += 1
            return stats

        stats.markets_scanned = len(groups)
        by_group = {g.group_id: g for g in groups}

        # Settle first: a resolved market frees a slot and returns real cash.
        self._settle_positions(stats)
        # Then take profit or cut losses on what is still open.
        if self.config.exits_enabled:
            self._run_exits(groups, stats)

        for group in groups:
            stats.groups_evaluated += 1
            if self._on_cooldown(group.group_id, now):
                continue
            try:
                fee = self._fee_model(group)
                signals = self.strategies.evaluate(group, fee)
            except Exception as exc:  # noqa: BLE001 - one bad market must not stop the loop
                log.warning("strategy error on %s: %s", group.title[:40], exc)
                stats.errors += 1
                continue

            theme = self._theme_for(group)
            held = self.portfolio.holds(group.group_id) if self.portfolio is not None else False
            for sig in signals:
                stats.signals_found += 1
                stats.best_edge = max(stats.best_edge, sig.edge_per_set)
                if theme and "theme" not in sig.metadata:
                    sig.metadata["theme"] = theme
                if self._chased(group.group_id, sig):
                    log.debug("anti-chase rejected %s", sig.describe())
                    stats.signals_rejected += 1
                    continue
                # Never open a second position on a market we already hold. The
                # cooldown above is only a timer and will always expire, so it
                # cannot be the thing that prevents doubling up.
                if held:
                    stats.signals_rejected += 1
                    log.debug("already holding %s; skipping %s", group.group_id[:16], sig.describe())
                    continue
                decision = self.risk.size_signal(
                    sig,
                    fee,
                    available_cash=self._available_cash(),
                    open_exposure=self._open_exposure(),
                    open_positions=self.portfolio.open_positions if self.portfolio else 0,
                    theme_exposure=self.portfolio.theme_exposure(theme) if self.portfolio else ZERO,
                )
                if not decision.approved:
                    stats.signals_rejected += 1
                    log.debug("risk rejected %s: %s", sig.describe(), decision.reason)
                    continue
                if not self.risk.rate_ok():
                    stats.signals_rejected += 1
                    log.warning("rate limit hit; skipping %s", sig.describe())
                    continue

                if sig.metadata.get("directional"):
                    log.warning(
                        "DIRECTIONAL trade (unhedged, model-driven edge): %s — %s",
                        sig.title[:50],
                        sig.metadata.get("warning", ""),
                    )

                result = self._execute(sig, decision.usd, group, fee, theme=theme)
                if result.ok:
                    stats.orders_filled += 1
                    stats.notional_usd += result.notional_usd
                    self._set_cooldown(group.group_id, now)
                    log.info("FILLED %s", result.describe())
                else:
                    stats.errors += 1
                    log.warning("NOT FILLED %s", result.describe())

        log.info("cycle: %s | portfolio: %s", stats.as_dict(), self.portfolio.summary() if self.portfolio else "n/a")
        return stats

    # ------------------------------------------------------------------ exits
    def _settle_positions(self, stats: CycleStats) -> None:
        """Close positions whose market has resolved, at terminal value.

        A basket position spans several condition ids (one per outcome), so
        settlement is only attempted once every leg's market has resolved.
        """
        if self.portfolio is None or not self.portfolio.positions:
            return
        wanted: list[str] = []
        for position in self.portfolio.positions:
            wanted.extend(self._condition_ids(position))
        if not wanted:
            return
        try:
            resolutions = self.scanner.fetch_resolutions(wanted)
        except Exception as exc:  # noqa: BLE001 - settlement is best effort
            log.debug("resolution fetch failed: %s", exc)
            return

        for position in list(self.portfolio.positions):
            condition_ids = self._condition_ids(position)
            if not condition_ids:
                continue
            resolved = [resolutions.get(c) for c in condition_ids]
            if any(r is None or not r.resolved for r in resolved):
                continue
            winners: list[str] = []
            for r in resolved:
                winners.extend(r.winning_token_ids)
            result = self.exits.settle(position, tuple(winners))
            stats.settled += 1
            stats.realized_usd += result.pnl_usd
            log.info(
                "SETTLED %s payout=$%.2f pnl=$%.2f — %s",
                position.title[:48],
                result.payout_usd,
                result.pnl_usd,
                "won" if result.winning else "lost",
            )

    @staticmethod
    def _condition_ids(position) -> list[str]:
        ids = [leg.condition_id for leg in position.legs if leg.condition_id]
        if ids:
            return list(dict.fromkeys(ids))
        return [position.condition_id] if position.condition_id else []

    def _priced_group(self, position) -> MarketGroup | None:
        """Rebuild a minimal group from a held position's own legs and books.

        Used only when the market is absent from the scan, so the position can
        still be valued and closed.
        """
        token_ids = [leg.token_id for leg in position.legs]
        if not token_ids or self.scanner is None:
            return None
        try:
            books = self.scanner.fetch_books(token_ids)
        except Exception as exc:  # noqa: BLE001 - missing books just defers the exit
            log.debug("book refetch failed for %s: %s", position.title[:40], exc)
            return None
        outcomes = tuple(
            Outcome(
                index=i,
                name=leg.outcome_name,
                token_id=leg.token_id,
                book=books.get(leg.token_id),
            )
            for i, leg in enumerate(position.legs)
        )
        if not any(o.book is not None for o in outcomes):
            return None
        return MarketGroup(
            group_id=position.group_id,
            title=position.title,
            outcomes=outcomes,
            seconds_to_end=0,
            is_binary=len(outcomes) == 2,
            tick_size=position.tick_size,
            neg_risk=position.neg_risk,
        )

    def _run_exits(self, groups: list[MarketGroup], stats: CycleStats) -> None:
        """Take profit or stop out on open positions, if the book allows."""
        if self.portfolio is None or not self.portfolio.positions:
            return
        by_group = {g.group_id: g for g in groups}
        for position in list(self.portfolio.positions):
            group = by_group.get(position.group_id)
            if group is None:
                # The market drifted out of the scan window (held markets can
                # fall below the volume cutoff). Fetch fresh books for its legs
                # rather than leaving the position stranded with no exit.
                group = self._priced_group(position)
                if group is None:
                    continue
            try:
                fee = self._fee_model(group)
                decision = self.exits.evaluate(position, group, fee)
            except Exception as exc:  # noqa: BLE001 - one bad exit must not stop the loop
                log.warning("exit evaluation failed for %s: %s", position.title[:40], exc)
                stats.errors += 1
                continue
            if not decision.should_exit:
                continue
            if not self.risk.rate_ok():
                log.debug("rate limit hit; deferring exit for %s", position.title[:40])
                continue
            log.info(
                "%s %s pnl=$%.2f (%s)",
                "STOP-LOSS" if decision.reason == "stop_loss" else "TAKE-PROFIT",
                position.title[:48],
                decision.pnl_usd,
                decision.detail,
            )
            if self.exits.close(position, group, decision):
                stats.exits += 1
                if decision.reason == "stop_loss":
                    stats.stop_losses += 1
                stats.realized_usd += decision.pnl_usd
            else:
                stats.errors += 1

    def _execute(
        self,
        signal: Signal,
        usd: Decimal,
        group: MarketGroup,
        fee: FeeModel,
        theme: str = "",
    ) -> ExecutionResult:
        try:
            return self.execution.execute(signal, usd, group, fee, theme=theme)
        except Exception as exc:  # noqa: BLE001 - never let execution kill the loop
            log.exception("execution crashed")
            return ExecutionResult(False, signal.kind, signal.title, error=str(exc))


    # ------------------------------------------------------------------- run
    def run_forever(self, max_cycles: int | None = None) -> None:
        interval = max(2, self.config.poll_interval_seconds)
        log.info(
            "starting bot: mode=%s interval=%ds strategies=%s",
            self.config.mode,
            interval,
            ", ".join(
                name
                for name, on in (
                    ("set_arb", self.config.arb_enabled),
                    ("basket_arb", self.config.basket_enabled),
                    ("fade", self.config.fade_enabled),
                )
                if on
            ),
        )
        if self.config.kill_switch:
            log.warning("kill switch is engaged: signals will be detected but never executed")

        self._install_signal_handlers()
        cycles = 0
        try:
            while not self._stopping:
                start = time.monotonic()
                try:
                    self.run_cycle()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.exception("cycle crashed: %s", exc)
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                elapsed = time.monotonic() - start
                self._sleep(max(0.0, interval - elapsed))
        except KeyboardInterrupt:
            log.info("interrupted by user")
        finally:
            self.shutdown()
            log.info("final portfolio: %s", self.portfolio.summary() if self.portfolio else "n/a")

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep, so Ctrl+C is responsive."""
        deadline = time.monotonic() + seconds
        while not self._stopping and time.monotonic() < deadline:
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            log.info("shutdown signal received")
            self._stopping = True

        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal_module, name, None)
            if sig is None:
                continue
            try:
                signal_module.signal(sig, handler)
            except (ValueError, OSError):
                # Not on the main thread; Ctrl+C still works via KeyboardInterrupt.
                pass

    def stop(self) -> None:
        self._stopping = True

    def shutdown(self) -> None:
        if self.portfolio is not None and self.config.state_file:
            try:
                self.portfolio.save(self.config.state_file)
                log.info("portfolio state saved to %s", self.config.state_file)
            except OSError as exc:
                log.warning("could not save state: %s", exc)
        closer = getattr(self.broker, "close", None)
        if callable(closer):
            closer()
