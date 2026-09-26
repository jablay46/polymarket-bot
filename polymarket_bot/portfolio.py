"""Portfolio state: cash, open exposure, positions, and realized PnL.

The paper portfolio is a real ledger, not a counter. It tracks how much cash
each signal consumes, how much of that is still at risk, and how much is
locked in a hedged complete set. Live mode reads balances from the venue; the
ledger is only used for paper mode and for reporting.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path

from .logging_setup import get_logger
from .models import ONE, ZERO, Signal

log = get_logger("portfolio")


@dataclass
class Position:
    """The legs bought together for one signal.

    ``hedged`` is True only for a complete set, where the payout at resolution
    is a known $1 per set. Directional positions (the fade strategy) carry an
    *assumed* exit price instead, so their profit is not comparable.
    """

    signal_kind: str
    group_id: str
    title: str
    token_ids: tuple[str, ...]
    shares: Decimal
    cost_usd: Decimal
    opened_at: float
    edge_per_set: Decimal
    expected_payout_usd: Decimal
    hedged: bool = True

    @property
    def expected_profit_usd(self) -> Decimal:
        return self.expected_payout_usd - self.cost_usd


@dataclass
class Portfolio:
    """Cash and position ledger for paper trading."""

    starting_balance: Decimal = Decimal("1000")
    cash: Decimal = field(init=False, default=ZERO)
    realized_pnl: Decimal = Decimal(0)
    positions: list[Position] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.starting_balance = Decimal(str(self.starting_balance))
        if self.cash == 0:
            self.cash = self.starting_balance

    # ------------------------------------------------------------ accounting
    @property
    def open_exposure(self) -> Decimal:
        """Cash currently committed to open positions."""
        return sum((p.cost_usd for p in self.positions), ZERO)

    @property
    def expected_payout(self) -> Decimal:
        """Certain payout, from hedged complete sets only."""
        return sum((p.expected_payout_usd for p in self.positions if p.hedged), ZERO)

    @property
    def expected_profit(self) -> Decimal:
        """Certain profit locked in by hedged complete sets."""
        return sum((p.expected_profit_usd for p in self.positions if p.hedged), ZERO)

    @property
    def directional_exposure(self) -> Decimal:
        """Cash at risk in unhedged, model-driven positions."""
        return sum((p.cost_usd for p in self.positions if not p.hedged), ZERO)

    @property
    def assumed_profit(self) -> Decimal:
        """Profit that depends on a directional position's exit assumption."""
        return sum((p.expected_profit_usd for p in self.positions if not p.hedged), ZERO)

    @property
    def equity(self) -> Decimal:
        return self.cash + self.open_exposure

    @property
    def open_positions(self) -> int:
        return len(self.positions)

    def available_cash(self) -> Decimal:
        return self.cash

    # -------------------------------------------------------------- mutation
    def open_position(self, signal: Signal, cost_usd: Decimal) -> Position:
        """Record a filled position and debit the cash it consumed.

        ``cost_usd`` is the cash actually spent on the fills. ``signal.max_sets``
        is the number of sets bought; a hedged set pays $1 at resolution.
        """
        cost_usd = Decimal(str(cost_usd))
        with self._lock:
            if cost_usd > self.cash:
                raise ValueError(f"insufficient cash: need {cost_usd}, have {self.cash}")
            self.cash -= cost_usd
            position = Position(
                signal_kind=signal.kind,
                group_id=signal.group_id,
                title=signal.title,
                token_ids=signal.token_ids,
                shares=signal.max_sets,
                cost_usd=cost_usd,
                opened_at=time.time(),
                edge_per_set=signal.edge_per_set,
                expected_payout_usd=signal.max_sets * signal.payout_per_set,
                hedged=not signal.metadata.get("directional", False),
            )
            self.positions.append(position)
            self.history.append(
                {
                    "event": "open",
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "kind": signal.kind,
                    "title": signal.title,
                    "group_id": signal.group_id,
                    "shares": str(position.shares),
                    "cost_usd": str(cost_usd),
                    "edge_per_set": str(signal.edge_per_set),
                    "expected_profit_usd": str(position.expected_profit_usd),
                }
            )
            return position

    def close_position(self, position: Position, proceeds_usd: Decimal, reason: str = "resolved") -> Decimal:
        """Close a position, crediting `proceeds_usd` back to cash."""
        with self._lock:
            proceeds = Decimal(str(proceeds_usd))
            if position in self.positions:
                self.positions.remove(position)
            self.cash += proceeds
            pnl = proceeds - position.cost_usd
            self.realized_pnl += pnl
            self.history.append(
                {
                    "event": "close",
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "kind": position.signal_kind,
                    "title": position.title,
                    "proceeds_usd": str(proceeds),
                    "cost_usd": str(position.cost_usd),
                    "pnl_usd": str(pnl),
                    "reason": reason,
                }
            )
            return pnl

    # ------------------------------------------------------------ persistence
    def snapshot(self) -> dict:
        return {
            "starting_balance": str(self.starting_balance),
            "cash": str(self.cash),
            "realized_pnl": str(self.realized_pnl),
            "open_exposure": str(self.open_exposure),
            "hedged_exposure": str(self.open_exposure - self.directional_exposure),
            "directional_exposure": str(self.directional_exposure),
            "expected_payout": str(self.expected_payout),
            "expected_profit": str(self.expected_profit),
            "assumed_profit": str(self.assumed_profit),
            "open_positions": self.open_positions,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "snapshot": self.snapshot(),
            "positions": [
                {**asdict(p), "shares": str(p.shares), "cost_usd": str(p.cost_usd),
                 "edge_per_set": str(p.edge_per_set), "expected_payout_usd": str(p.expected_payout_usd),
                 "token_ids": list(p.token_ids)}
                for p in self.positions
            ],
            "history": self.history[-500:],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def summary(self) -> str:
        return (
            f"cash=${self.cash:.2f} exposure=${self.open_exposure:.2f} "
            f"equity=${self.equity:.2f} open={self.open_positions} "
            f"realized=${self.realized_pnl:.2f} "
            f"hedged_profit=${self.expected_profit:.2f} "
            f"directional=${self.directional_exposure:.2f} (assumed_profit=${self.assumed_profit:.2f})"
        )
