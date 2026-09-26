"""Execution: turn an approved signal into placed orders.

The important property here is *leg-risk containment*. A multi-leg arbitrage
is only risk-free if every leg fills. If a later leg fails after an earlier one
filled, the position is no longer hedged. To keep that exposure bounded:

* legs are submitted in descending notional order, so the largest and least
  certain leg goes first;
* the requested notional is scaled down to what the smallest leg can support;
* if any leg fails, the filled legs are unwound immediately at the market;
* a failed unwind is logged loudly and surfaced in the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .brokers import BrokerError, LegResult, PaperBroker
from .config import Config
from .fees import FeeModel
from .logging_setup import get_logger
from .models import ONE, ZERO, MarketGroup, Signal, quantize_down
from .portfolio import Portfolio

log = get_logger("execution")


@dataclass
class ExecutionResult:
    ok: bool
    kind: str = ""
    title: str = ""
    notional_usd: Decimal = ZERO
    legs: list[LegResult] = field(default_factory=list)
    error: str = ""
    unwound: bool = False

    @property
    def status(self) -> str:
        if self.ok:
            return "filled"
        if self.error:
            return "error"
        return "rejected"

    def describe(self) -> str:
        if self.ok:
            return f"filled {len(self.legs)} leg(s) ${self.notional_usd:.2f} — {self.title[:48]}"
        return f"{self.status}: {self.error or 'unknown'} — {self.title[:48]}"


@dataclass
class ExecutionEngine:
    """Submit signals through a broker and maintain the portfolio ledger."""

    config: Config
    broker: object
    portfolio: Portfolio | None = None

    def __post_init__(self) -> None:
        self.orders_submitted = 0
        self.orders_failed = 0
        self.leg_risk_events = 0

    # ------------------------------------------------------------ entrypoint
    def execute(self, signal: Signal, notional_usd: Decimal, group: MarketGroup, fee: FeeModel) -> ExecutionResult:
        if notional_usd <= 0:
            return ExecutionResult(False, signal.kind, signal.title, error="non-positive notional")
        if signal.max_sets <= 0:
            return ExecutionResult(False, signal.kind, signal.title, error="no size")

        # Scale legs to the approved notional. Sizing already bounded this by
        # the smallest leg's depth, so this is a proportional trim.
        scale = min(ONE, notional_usd / signal.notional_usd) if signal.notional_usd > 0 else ZERO
        if scale <= 0:
            return ExecutionResult(False, signal.kind, signal.title, error="notional too small")

        plan: list[tuple[str, Decimal, Decimal]] = []
        for leg in signal.legs:
            shares = quantize_down(leg.shares * scale, Decimal("0.01"))
            usd = shares * leg.price
            if shares <= 0 or usd <= 0:
                return ExecutionResult(False, signal.kind, signal.title, error=f"leg {leg.outcome_name} rounds to zero")
            plan.append((leg.token_id, shares, leg.price))
        # Largest first, so the least certain leg commits the most capital.
        plan.sort(key=lambda item: item[1] * item[2], reverse=True)

        tick = group.tick_size
        neg_risk = group.neg_risk
        results: list[LegResult] = []
        total_usd = ZERO

        for token_id, shares, price in plan:
            usd = shares * price
            result = self.broker.buy(
                token_id=token_id,
                usd=usd,
                price=price,
                tick_size=tick,
                neg_risk=neg_risk,
                order_type=self.config.live_order_type,
            )
            results.append(result)
            if result.ok:
                total_usd += result.filled_usd if result.filled_usd > 0 else usd
                self.orders_submitted += 1
            else:
                self.orders_failed += 1
                log.warning("leg failed for %s: %s", signal.title[:40], result.error)
                unwound = self._unwind(results, tick, neg_risk)
                if not unwound:
                    self.leg_risk_events += 1
                    log.error(
                        "LEG RISK: partial fill could not be unwound for %s — manual review required",
                        signal.title[:60],
                    )
                return ExecutionResult(
                    False,
                    signal.kind,
                    signal.title,
                    notional_usd=total_usd,
                    legs=results,
                    error=f"leg failed: {result.error}",
                    unwound=unwound,
                )

        # Success. Record the position for reporting and paper accounting.
        if self.portfolio is not None:
            try:
                self.portfolio.open_position(signal, total_usd)
            except ValueError as exc:
                log.warning("could not record position: %s", exc)
        return ExecutionResult(True, signal.kind, signal.title, notional_usd=total_usd, legs=results)

    def _unwind(self, results: list[LegResult], tick_size: Decimal, neg_risk: bool) -> bool:
        """Sell back every leg that filled, best effort."""
        filled = [r for r in results if r.ok and r.filled_shares > 0]
        if not filled:
            return True
        all_ok = True
        for result in filled:
            price = self._unwind_price(result)
            sell = self.broker.sell(
                token_id=result.token_id,
                shares=result.filled_shares,
                price=price,
                tick_size=tick_size,
                neg_risk=neg_risk,
                order_type="FAK",
            )
            if not sell.ok:
                all_ok = False
                log.error("unwind sell failed for %s: %s", result.token_id[:12], sell.error)
        return all_ok

    @staticmethod
    def _unwind_price(result: LegResult) -> Decimal:
        if result.filled_shares > 0 and result.filled_usd > 0:
            return result.filled_usd / result.filled_shares
        return ZERO
