"""Risk management: sizing, exposure limits, rate limits, kill switch.

Sizing is bounded by independent constraints and takes the smallest:

1. Per-order notional (``max_order_usd``).
2. Remaining portfolio exposure (``max_total_exposure_usd``).
3. Available cash (paper balance or live wallet balance).
4. Book impact (``max_book_impact``): a size is shrunk until its average fill
   price is within that fraction of the best ask, so we never quote far through
   the book chasing a nominal edge.
5. Concurrent open positions (``max_open_positions``).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal

from .config import Config
from .fees import FeeModel
from .logging_setup import get_logger
from .models import ONE, ZERO, Signal, quantize_down

log = get_logger("risk")


@dataclass
class RiskDecision:
    approved: bool
    usd: Decimal = ZERO
    reason: str = ""


@dataclass
class RiskManager:
    """Gate every order through sizing, exposure, and rate limits."""

    config: Config
    _timestamps: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    orders_placed: int = 0
    orders_blocked: int = 0

    # ------------------------------------------------------------- switches
    @property
    def kill_switch(self) -> bool:
        return bool(self.config.kill_switch)

    def rate_ok(self) -> bool:
        """Token-bucket style limiter over a rolling one-minute window."""
        now = time.monotonic()
        with self._lock:
            while self._timestamps and now - self._timestamps[0] > 60:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.config.max_orders_per_minute:
                return False
            self._timestamps.append(now)
            return True

    def note_order(self) -> None:
        self.orders_placed += 1

    # --------------------------------------------------------------- sizing
    def size_signal(
        self,
        signal: Signal,
        fee: FeeModel,
        *,
        available_cash: Decimal,
        open_exposure: Decimal,
        open_positions: int = 0,
        theme_exposure: Decimal = ZERO,
    ) -> RiskDecision:
        """Return the approved order notional for a signal, or a rejection."""
        if self.kill_switch:
            return RiskDecision(False, reason="kill switch engaged")
        if signal.max_sets <= 0:
            return RiskDecision(False, reason="no book depth")
        if open_positions >= self.config.max_open_positions:
            return RiskDecision(False, reason="max open positions reached")

        per_order = Decimal(str(self.config.max_order_usd))
        exposure_headroom = Decimal(str(self.config.max_total_exposure_usd)) - open_exposure
        if exposure_headroom <= 0:
            return RiskDecision(False, reason="total exposure limit reached")

        # Correlated markets (several "Iran by September" questions) are one
        # bet, not several, so they share a tighter budget than the portfolio
        # as a whole. Signals with no resolved theme are not capped here; the
        # total-exposure limit already bounds them.
        theme_cap = Decimal(str(self.config.max_theme_exposure_usd))
        theme = signal.metadata.get("theme") or ""
        if theme_cap > 0 and theme:
            theme_headroom = theme_cap - theme_exposure
            if theme_headroom <= 0:
                return RiskDecision(False, reason=f"theme exposure limit reached ({theme})")
        else:
            theme_headroom = exposure_headroom

        cash_headroom = available_cash - Decimal(str(self.config.min_free_balance_usd))
        if cash_headroom <= 0:
            return RiskDecision(False, reason="insufficient free balance")

        # What does one complete set cost, including taker fees?
        cost_per_set = signal.cost_per_set
        if cost_per_set <= 0:
            return RiskDecision(False, reason="invalid set cost")
        fee_per_set = self._fee_per_set(signal, fee)
        all_in_per_set = cost_per_set + fee_per_set

        budget = min(per_order, exposure_headroom, cash_headroom, theme_headroom)
        sets = quantize_down(budget / all_in_per_set, ONE)
        sets = min(sets, signal.max_sets)
        sets = self._apply_book_impact_cap(signal, sets)
        if sets <= 0:
            # Nothing affordable at a whole share. Report which limit bit.
            if budget < all_in_per_set:
                reason = f"budget ${budget:.2f} below one set at ${all_in_per_set:.4f}"
            else:
                reason = "book impact cap leaves no size"
            return RiskDecision(False, reason=reason)

        usd = sets * all_in_per_set
        if usd <= 0:
            return RiskDecision(False, reason="zero notional")
        return RiskDecision(True, usd=usd, reason="ok")

    def _apply_book_impact_cap(self, signal: Signal, sets: Decimal) -> Decimal:
        """Shrink a size until no leg's average fill strays from its touch price.

        Average fill price is non-decreasing in size, so if the reported size is
        within the cap every smaller size is too. Otherwise the size is scaled
        down proportionally to the worst offender, which is a safe under-estimate
        because the price curve is concave near the top of the book.
        """
        cap = Decimal(str(self.config.max_book_impact))
        if cap <= 0 or sets <= 0 or not signal.legs:
            return sets
        worst = max((leg.book_impact for leg in signal.legs), default=ZERO)
        if worst <= cap:
            return sets
        scaled = quantize_down(sets * (cap / worst), ONE)
        if scaled < 1:
            return ZERO
        return min(scaled, sets)

    @staticmethod
    def _fee_per_set(signal: Signal, fee: FeeModel) -> Decimal:
        if fee.is_fee_free or not signal.legs:
            return ZERO
        total = ZERO
        for leg in signal.legs:
            total += fee.buy_fee(leg.shares, leg.price)
        return total / signal.max_sets if signal.max_sets > 0 else ZERO

    def can_trade(self, exposure_usd: Decimal, available_cash: Decimal, open_positions: int = 0) -> bool:
        if self.kill_switch:
            return False
        if open_positions >= self.config.max_open_positions:
            return False
        if exposure_usd >= Decimal(str(self.config.max_total_exposure_usd)):
            return False
        if available_cash <= Decimal(str(self.config.min_free_balance_usd)):
            return False
        return True

