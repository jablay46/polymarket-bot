"""Domain models: order books, markets, outcome sets, and trade signals.

Prices and sizes use :class:`decimal.Decimal` throughout so that cent-level
arithmetic stays exact.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable

ZERO = Decimal(0)
ONE = Decimal(1)


def to_decimal(value: Any, default: Decimal = ZERO) -> Decimal:
    """Best-effort conversion to Decimal for API values of mixed types."""
    if value is None or value == "":
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return default


def seconds_until(raw: Any) -> float | None:
    """Seconds from now until an ISO-8601 timestamp, or None if unparseable."""
    if not raw:
        return None
    if isinstance(raw, dt.datetime):
        end = raw
    else:
        text = str(raw).strip().replace("Z", "+00:00")
        try:
            end = dt.datetime.fromisoformat(text)
        except ValueError:
            return None
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)
    return (end - dt.datetime.now(dt.timezone.utc)).total_seconds()


@dataclass(frozen=True)
class PriceLevel:
    price: Decimal
    size: Decimal

    @property
    def notional(self) -> Decimal:
        return self.price * self.size


@dataclass(frozen=True)
class FillEstimate:
    """Result of walking the book for a requested size or budget."""

    shares: Decimal
    usd: Decimal
    worst_price: Decimal
    levels_used: int
    requested: Decimal = ZERO

    @property
    def avg_price(self) -> Decimal:
        if self.shares <= 0:
            return ZERO
        return self.usd / self.shares

    @property
    def complete(self) -> bool:
        """True when the request was fully satisfied.

        For a share request this means every requested share was filled. For a
        budget request (``requested`` is zero) it means at least some shares
        were bought.
        """
        if self.requested > 0:
            return self.shares >= self.requested
        return self.shares > 0


@dataclass(frozen=True)
class OrderBook:
    """A normalized CLOB order book.

    `bids` is sorted best-first (highest price first) and `asks` best-first
    (lowest price first). Normalizing on parse means callers never depend on
    the venue's array ordering, which is not guaranteed.
    """

    token_id: str
    bids: tuple[PriceLevel, ...] = ()
    asks: tuple[PriceLevel, ...] = ()
    tick_size: Decimal = Decimal("0.01")
    min_order_size: Decimal = Decimal("5")
    neg_risk: bool = False

    # ------------------------------------------------------------ construction
    @classmethod
    def from_api(cls, payload: dict, token_id: str = "") -> "OrderBook":
        """Build from a raw CLOB `/book` response."""
        if not payload:
            raise ValueError("empty order book payload")

        def levels(key: str, best_first_desc: bool) -> tuple[PriceLevel, ...]:
            raw = payload.get(key) or []
            parsed = []
            for item in raw:
                if isinstance(item, dict):
                    price, size = item.get("price"), item.get("size")
                else:
                    price, size = item[0], item[1]
                p = to_decimal(price)
                s = to_decimal(size)
                if p > 0 and s > 0:
                    parsed.append(PriceLevel(p, s))
            parsed.sort(key=lambda lvl: lvl.price, reverse=best_first_desc)
            return tuple(parsed)

        return cls(
            token_id=str(payload.get("asset_id") or token_id),
            bids=levels("bids", True),
            asks=levels("asks", False),
            tick_size=to_decimal(payload.get("tick_size"), Decimal("0.01")),
            min_order_size=to_decimal(payload.get("min_order_size"), Decimal("5")),
            neg_risk=bool(payload.get("neg_risk", False)),
        )

    # ---------------------------------------------------------------- summary
    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return bid if bid is not None else ask

    @property
    def spread(self) -> Decimal | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def top_ask_size(self) -> Decimal:
        return self.asks[0].size if self.asks else ZERO

    @property
    def top_bid_size(self) -> Decimal:
        return self.bids[0].size if self.bids else ZERO

    def has_two_sided_market(self) -> bool:
        return bool(self.bids) and bool(self.asks)

    # ------------------------------------------------------------- book walks
    def cost_to_buy(self, shares: Decimal, price_limit: Decimal | None = None) -> FillEstimate:
        """Walk asks ascending, spending at most `price_limit` per share."""
        remaining = shares
        usd = ZERO
        worst = ZERO
        used = 0
        for level in self.asks:
            if remaining <= 0:
                break
            if price_limit is not None and level.price > price_limit:
                break
            take = min(level.size, remaining)
            usd += take * level.price
            remaining -= take
            worst = level.price
            used += 1
        filled = shares - remaining
        return FillEstimate(filled, usd, worst, used, requested=shares)

    def shares_for_budget(self, budget: Decimal, price_limit: Decimal | None = None) -> FillEstimate:
        """Walk asks ascending, buying as many shares as `budget` allows."""
        if budget <= 0:
            return FillEstimate(ZERO, ZERO, ZERO, 0)
        spent = ZERO
        shares = ZERO
        worst = ZERO
        used = 0
        for level in self.asks:
            if price_limit is not None and level.price > price_limit:
                break
            cost = level.size * level.price
            if spent + cost <= budget:
                spent += cost
                shares += level.size
                worst = level.price
                used += 1
                continue
            affordable = (budget - spent) / level.price
            if affordable > 0:
                shares += affordable
                spent += affordable * level.price
                worst = level.price
                used += 1
            break
        return FillEstimate(shares, spent, worst, used)

    def proceeds_to_sell(self, shares: Decimal, price_floor: Decimal | None = None) -> FillEstimate:
        """Walk bids descending, selling at least `price_floor` per share."""
        remaining = shares
        usd = ZERO
        worst = ZERO
        used = 0
        for level in self.bids:
            if remaining <= 0:
                break
            if price_floor is not None and level.price < price_floor:
                break
            take = min(level.size, remaining)
            usd += take * level.price
            remaining -= take
            worst = level.price
            used += 1
        filled = shares - remaining
        return FillEstimate(filled, usd, worst, used, requested=shares)

    def depth_usd(self, side: str = "ask", levels: int = 3) -> Decimal:
        book = self.asks if side == "ask" else self.bids
        return sum((lvl.notional for lvl in book[:levels]), ZERO)


@dataclass(frozen=True)
class Outcome:
    """One tradable outcome token inside a market."""

    index: int
    name: str
    token_id: str
    book: OrderBook | None = None

    @property
    def best_ask(self) -> Decimal | None:
        return self.book.best_ask if self.book else None

    @property
    def best_bid(self) -> Decimal | None:
        return self.book.best_bid if self.book else None


@dataclass(frozen=True)
class MarketGroup:
    """A set of outcomes that partition the event space.

    A binary market is a two-outcome group. A neg-risk event (for example
    "EPL 2027 Champion") is an N-outcome group whose Yes tokens sum to
    approximately 1, because exactly one outcome can resolve true.
    """

    group_id: str
    title: str
    outcomes: tuple[Outcome, ...]
    volume_24h: Decimal = ZERO
    liquidity: Decimal = ZERO
    seconds_to_end: float | None = None
    neg_risk: bool = False
    is_binary: bool = True
    fees_enabled: bool = False
    tick_size: Decimal = Decimal("0.01")
    metadata: dict = field(default_factory=dict)

    def with_books(self, books: dict[str, OrderBook]) -> "MarketGroup":
        outcomes = tuple(
            Outcome(o.index, o.name, o.token_id, books.get(o.token_id, o.book)) for o in self.outcomes
        )
        return MarketGroup(
            group_id=self.group_id,
            title=self.title,
            outcomes=outcomes,
            volume_24h=self.volume_24h,
            liquidity=self.liquidity,
            seconds_to_end=self.seconds_to_end,
            neg_risk=self.neg_risk,
            is_binary=self.is_binary,
            fees_enabled=self.fees_enabled,
            tick_size=self.tick_size,
            metadata=dict(self.metadata),
        )

    @property
    def n_outcomes(self) -> int:
        return len(self.outcomes)

    def with_books_present(self) -> bool:
        return all(o.book is not None for o in self.outcomes)

    def ask_sum(self) -> Decimal | None:
        prices = [o.best_ask for o in self.outcomes]
        if any(p is None for p in prices):
            return None
        return sum(prices, ZERO)

    def bid_sum(self) -> Decimal | None:
        prices = [o.best_bid for o in self.outcomes]
        if any(p is None for p in prices):
            return None
        return sum(prices, ZERO)

    def mid_sum(self) -> Decimal | None:
        mids = [o.book.mid if o.book else None for o in self.outcomes]
        if any(m is None for m in mids):
            return None
        return sum(mids, ZERO)


@dataclass(frozen=True)
class Leg:
    """A single order within a signal."""

    token_id: str
    outcome_name: str
    price: Decimal
    shares: Decimal
    usd: Decimal
    side: str = "BUY"
    # Best price available on this side when the signal was built. The gap
    # between this and `price` (the average fill) is the book impact.
    touch_price: Decimal = ZERO

    @property
    def book_impact(self) -> Decimal:
        """Fractional slippage of the average fill against the touch price."""
        if self.touch_price <= 0 or self.price <= 0:
            return ZERO
        return (self.price - self.touch_price) / self.touch_price


@dataclass(frozen=True)
class Signal:
    """An actionable, fully specified trade opportunity."""

    kind: str
    group_id: str
    title: str
    legs: tuple[Leg, ...]
    edge_per_set: Decimal
    cost_per_set: Decimal
    payout_per_set: Decimal
    confidence: float
    expected_profit_usd: Decimal = ZERO
    max_sets: Decimal = ZERO
    metadata: dict = field(default_factory=dict)

    @property
    def notional_usd(self) -> Decimal:
        return sum((leg.usd for leg in self.legs), ZERO)

    @property
    def total_shares(self) -> Decimal:
        return sum((leg.shares for leg in self.legs), ZERO)

    @property
    def token_ids(self) -> tuple[str, ...]:
        return tuple(leg.token_id for leg in self.legs)

    def describe(self) -> str:
        legs = ", ".join(f"{l.outcome_name}@{l.price}" for l in self.legs)
        return (
            f"[{self.kind}] {self.title[:48]} edge/set={self.edge_per_set:.4f} "
            f"cost/set={self.cost_per_set:.4f} notional=${self.notional_usd:.2f} ({legs})"
        )



def quantize_down(value: Decimal, step: Decimal) -> Decimal:
    """Round `value` down to the nearest multiple of `step`."""
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding="ROUND_DOWN") * step




