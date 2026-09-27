"""Market data: Gamma discovery, CLOB books, and normalized market groups.

The CLOB returns ``asks`` sorted high-to-low and ``bids`` sorted low-to-high.
That means the first element of each array is the *worst* price, not the best.
:meth:`OrderBook.from_api` re-sorts both sides so ``best_ask``/``best_bid`` are
always the true top of book. Nothing downstream depends on venue ordering.
"""

from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from decimal import Decimal

from .config import Config
from .fees import resolve_taker_rate
from .logging_setup import get_logger
from .models import (
    MarketGroup,
    OrderBook,
    Outcome,
    ZERO,
    seconds_until,
    to_decimal,
)

log = get_logger("data")

_BATCH_LIMIT = 500

# Tags that describe a whole category rather than a shared story. Using one of
# these as a theme would lump unrelated markets into a single risk bucket.
_BROAD_TAGS = frozenset(
    {
        "all",
        "sports",
        "games",
        "politics",
        "geopolitics",
        "world",
        "crypto",
        "economy",
        "elections",
        "recurring",
        "culture",
        "tech",
        "business",
        "news",
    }
)


class DataError(RuntimeError):
    """Raised when a market data request cannot be completed."""


# --------------------------------------------------------------------- HTTP


class HttpClient:
    """Minimal JSON HTTP client with retries and jittered backoff."""

    def __init__(self, timeout: float = 10.0, max_retries: int = 3, user_agent: str = "polymarket-bot/1.0"):
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.user_agent = user_agent

    def _request(self, url: str, data: bytes | None = None, method: str = "GET") -> object:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("Accept", "application/json")
            req.add_header("User-Agent", self.user_agent)
            if data is not None:
                req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                return json.loads(body) if body else None
            except urllib.error.HTTPError as exc:
                # 4xx other than 429 will not improve on retry.
                if exc.code < 500 and exc.code != 429:
                    raise DataError(f"HTTP {exc.code} for {url}") from exc
                last_error = exc
            except Exception as exc:  # noqa: BLE001 - network layer, retry everything
                last_error = exc
            if attempt < self.max_retries - 1:
                time.sleep((0.4 * 2**attempt) + random.random() * 0.2)
        raise DataError(f"request failed after {self.max_retries} attempts: {url}: {last_error}")

    def get(self, url: str, params: dict | None = None) -> object:
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            # Gamma needs repeated keys for multi-value filters; comma-joining
            # them silently matches nothing.
            url = f"{url}?{urllib.parse.urlencode(clean, doseq=True)}"
        return self._request(url)

    def post(self, url: str, payload: object) -> object:
        return self._request(url, json.dumps(payload).encode("utf-8"), method="POST")


# ------------------------------------------------------------------ parsing


def _parse_json_field(value, default):
    """Gamma returns several fields as JSON-encoded strings."""
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        if isinstance(value, str) and "," in value:
            return [part.strip() for part in value.split(",")]
        return default


@dataclass
class MarketInfo:
    """Market metadata extracted from a Gamma market object."""

    market_id: str
    condition_id: str
    question: str
    slug: str
    token_ids: tuple[str, ...]
    outcomes: tuple[str, ...]
    volume_24h: object
    liquidity: object
    seconds_to_end: float | None
    neg_risk: bool
    fees_enabled: bool
    fee_type: str | None
    tick_size: object
    min_order_size: object
    group_item_title: str = ""
    neg_risk_market_id: str = ""
    event_id: str = ""
    event_title: str = ""
    accepting_orders: bool = True
    enable_order_book: bool = True

    @property
    def is_binary(self) -> bool:
        return len(self.token_ids) == 2

    @property
    def yes_index(self) -> int | None:
        """Index of the Yes token, matched by outcome name rather than position."""
        for i, name in enumerate(self.outcomes):
            if str(name).strip().lower() == "yes":
                return i
        return None


def parse_market(raw: dict, event: dict | None = None) -> MarketInfo | None:
    """Normalize one Gamma market payload.

    Gamma embeds the parent event under ``events`` for markets fetched from
    ``/markets``; that is the only place the event id and title appear, so it
    is used when no explicit event was passed. Themes and neg-risk grouping
    both depend on that id.
    """
    if event is None:
        embedded = raw.get("events") or []
        if embedded and isinstance(embedded[0], dict):
            event = embedded[0]
    if not raw:
        return None
    token_ids = _parse_json_field(raw.get("clobTokenIds"), [])
    outcomes = _parse_json_field(raw.get("outcomes"), [])
    if not token_ids:
        return None
    return MarketInfo(
        market_id=str(raw.get("id") or ""),
        condition_id=str(raw.get("conditionId") or ""),
        question=str(raw.get("question") or ""),
        slug=str(raw.get("slug") or ""),
        token_ids=tuple(str(t) for t in token_ids),
        outcomes=tuple(str(o) for o in outcomes),
        volume_24h=to_decimal(raw.get("volume24hr") or raw.get("volume24h")),
        liquidity=to_decimal(raw.get("liquidityNum") if raw.get("liquidityNum") is not None else raw.get("liquidity")),
        seconds_to_end=seconds_until(raw.get("endDate") or raw.get("endDateIso")),
        neg_risk=bool(raw.get("negRisk", False)),
        fees_enabled=bool(raw.get("feesEnabled", False)),
        fee_type=raw.get("feeType"),
        tick_size=to_decimal(raw.get("orderPriceMinTickSize"), to_decimal("0.01")),
        min_order_size=to_decimal(raw.get("orderMinSize"), to_decimal("5")),
        group_item_title=str(raw.get("groupItemTitle") or ""),
        neg_risk_market_id=str(raw.get("negRiskMarketID") or ""),
        event_id=str((event or {}).get("id") or ""),
        event_title=str((event or {}).get("title") or ""),
        accepting_orders=bool(raw.get("acceptingOrders", True)),
        enable_order_book=bool(raw.get("enableOrderBook", True)),
    )


@dataclass(frozen=True)
class Resolution:
    """The settled outcome of a market.

    ``winning_token_ids`` holds the token(s) that pay $1. A binary market has
    exactly one; a multi-outcome set can in principle have several.
    """

    condition_id: str
    resolved: bool
    winning_token_ids: tuple[str, ...] = ()
    outcome_prices: tuple[Decimal, ...] = ()
    token_ids: tuple[str, ...] = ()
    closed: bool = False
    raw_status: str = ""


def parse_resolution(raw: dict) -> Resolution | None:
    """Build a :class:`Resolution` from a Gamma market object.

    A market is treated as settled only when it is closed *and* its prices have
    snapped to a winner. ``umaResolutionStatus`` is recorded but not required,
    because it is absent on some older markets.
    """
    if not raw:
        return None
    token_ids = tuple(str(t) for t in _parse_json_field(raw.get("clobTokenIds"), []))
    prices_raw = _parse_json_field(raw.get("outcomePrices"), [])
    prices = tuple(to_decimal(p) for p in prices_raw)
    status = str(raw.get("umaResolutionStatus") or "")
    closed = bool(raw.get("closed", False))

    winners: list[str] = []
    if prices and len(prices) == len(token_ids):
        for token, price in zip(token_ids, prices):
            if price >= Decimal("0.99"):
                winners.append(token)
    resolved = closed and bool(winners)
    return Resolution(
        condition_id=str(raw.get("conditionId") or ""),
        resolved=resolved,
        winning_token_ids=tuple(winners),
        outcome_prices=prices,
        token_ids=token_ids,
        closed=closed,
        raw_status=status,
    )


# ------------------------------------------------------------------- scanner


@dataclass
class ScannerStats:
    """Counters for the most recent scan.

    Reset at the start of every :meth:`MarketScanner.scan`, so the "scan
    complete" log describes the cycle that just ran rather than a running
    total that only ever grows.
    """

    markets_seen: int = 0
    groups_built: int = 0
    books_fetched: int = 0
    books_missing: int = 0
    errors: int = 0

    def reset(self) -> None:
        self.markets_seen = 0
        self.groups_built = 0
        self.books_fetched = 0
        self.books_missing = 0
        self.errors = 0


@dataclass
class MarketScanner:
    """Discover markets and attach normalized order books."""

    config: Config
    http: HttpClient = None  # type: ignore[assignment]
    stats: ScannerStats = field(default_factory=ScannerStats)
    _fee_cache: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.http is None:
            self.http = HttpClient(
                timeout=self.config.request_timeout, max_retries=self.config.max_retries
            )

    # ---------------------------------------------------------- discovery
    def fetch_binary_markets(self, limit: int | None = None, pages: int = 1) -> list[MarketInfo]:
        """Highest 24h-volume active markets, as binary candidates.

        Gamma caps a page at 100 rows, so ``pages`` walks the offset. The
        cross-market ladder scan needs the deeper reach: crypto threshold
        markets sit well below the top of the volume ranking.
        """
        page_size = 100
        pages = max(1, pages)
        limit = limit if limit is not None else page_size * pages
        out: list[MarketInfo] = []
        seen: set[str] = set()
        for page in range(pages):
            raw = self.http.get(
                f"{self.config.gamma_host}/markets",
                {
                    "limit": page_size,
                    "offset": page * page_size,
                    "active": "true",
                    "closed": "false",
                    "order": "volume24hr",
                    "ascending": "false",
                },
            )
            if not raw:
                break
            for item in raw:
                info = parse_market(item)
                if info is None or info.market_id in seen:
                    continue
                seen.add(info.market_id)
                out.append(info)
            if len(raw) < page_size:
                break
        self.stats.markets_seen += len(out)
        return out[:limit]

    def fetch_neg_risk_events(self, limit: int | None = None) -> list[list[MarketInfo]]:
        """Multi-outcome neg-risk events, returned as candidate outcome sets."""
        limit = limit or max(10, self.config.scan_limit // 2)
        raw = self.http.get(
            f"{self.config.gamma_host}/events",
            {
                "limit": limit,
                "active": "true",
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
            },
        )
        groups: list[list[MarketInfo]] = []
        for event in raw or []:
            if not event.get("negRisk"):
                continue
            markets = event.get("markets") or []
            infos = []
            for item in markets:
                info = parse_market(item, event)
                if info is None or not info.is_binary or not info.enable_order_book:
                    continue
                if not info.accepting_orders:
                    continue
                if not info.group_item_title:
                    continue
                infos.append(info)
            if len(infos) >= self.config.basket_min_outcomes:
                groups.append(infos)
        return groups

    # --------------------------------------------------------------- themes
    def fetch_themes(self, event_ids: list[str]) -> dict[str, str]:
        """Map event id -> correlation theme, from the event's most specific tag.

        Broad tags like ``sports`` group hundreds of unrelated markets, so the
        rarest tag on the event is used: it is the one that actually describes
        the shared story ("iran", "strait-of-hormuz").
        """
        unique = [e for e in dict.fromkeys(event_ids) if e]
        if not unique:
            return {}
        events: list[dict] = []
        for start in range(0, len(unique), _BATCH_LIMIT):
            chunk = unique[start : start + _BATCH_LIMIT]
            try:
                raw = self.http.get(f"{self.config.gamma_host}/events", {"id": chunk})
            except DataError as exc:
                log.debug("theme lookup failed: %s", exc)
                continue
            events.extend(e for e in (raw or []) if isinstance(e, dict))

        # Count how many of *our* events carry each tag, then prefer the tag
        # that is most specific to this batch.
        tag_counts: dict[str, int] = {}
        per_event: list[tuple[str, list[str]]] = []
        for event in events:
            tags = [
                str(t.get("slug"))
                for t in (event.get("tags") or [])
                if isinstance(t, dict) and t.get("slug")
            ]
            tags = [t for t in tags if t not in _BROAD_TAGS]
            per_event.append((str(event.get("id") or ""), tags))
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        out: dict[str, str] = {}
        for event_id, tags in per_event:
            if not event_id or not tags:
                continue
            # Rarest tag wins; ties broken by name for determinism.
            out[event_id] = min(tags, key=lambda t: (tag_counts[t], t))
        return out

    # ---------------------------------------------------------- resolutions
    def fetch_resolutions(self, condition_ids: list[str]) -> dict[str, Resolution]:
        """Look up settlement state for markets we hold.

        Gamma hides closed markets by default, so ``closed=true`` must be sent
        explicitly or every settled market silently reads back as unknown.
        """
        unique = [c for c in dict.fromkeys(condition_ids) if c]
        if not unique:
            return {}
        out: dict[str, Resolution] = {}
        for start in range(0, len(unique), _BATCH_LIMIT):
            chunk = unique[start : start + _BATCH_LIMIT]
            try:
                raw = self.http.get(
                    f"{self.config.gamma_host}/markets",
                    {"condition_ids": chunk, "closed": "true", "limit": len(chunk)},
                )
            except DataError as exc:
                log.warning("resolution lookup failed: %s", exc)
                continue
            for item in raw or []:
                res = parse_resolution(item)
                if res is not None and res.condition_id:
                    out[res.condition_id] = res
        return out

    # --------------------------------------------------------------- books
    def fetch_books(self, token_ids: list[str]) -> dict[str, OrderBook]:
        """Fetch many order books, batching where possible."""
        unique = [t for t in dict.fromkeys(token_ids) if t]
        books: dict[str, OrderBook] = {}
        for start in range(0, len(unique), _BATCH_LIMIT):
            chunk = unique[start : start + _BATCH_LIMIT]
            try:
                payload = self.http.post(
                    f"{self.config.clob_host}/books",
                    [{"token_id": tid} for tid in chunk],
                )
            except DataError as exc:
                log.debug("batch book fetch failed (%s), falling back to per-token", exc)
                payload = None
            if isinstance(payload, list):
                for entry in payload:
                    if not entry:
                        continue
                    book = self._build_book(entry)
                    if book is not None:
                        books[book.token_id] = book
            else:
                for tid in chunk:
                    book = self.fetch_book(tid)
                    if book is not None:
                        books[tid] = book
        missing = [t for t in unique if t not in books]
        self.stats.books_fetched += len(books)
        self.stats.books_missing += len(missing)
        return books

    def fetch_book(self, token_id: str) -> OrderBook | None:
        try:
            payload = self.http.get(f"{self.config.clob_host}/book", {"token_id": token_id})
        except DataError as exc:
            log.debug("book fetch failed for %s: %s", token_id[:12], exc)
            return None
        return self._build_book(payload)

    @staticmethod
    def _build_book(payload: dict | None) -> OrderBook | None:
        if not payload:
            return None
        try:
            return OrderBook.from_api(payload)
        except (ValueError, KeyError, TypeError) as exc:
            log.debug("malformed book: %s", exc)
            return None

    # ---------------------------------------------------------------- fees
    def fee_rate_for(self, info: MarketInfo) -> float:
        """Category taker fee rate for a market, cached per fee type."""
        if self.config.taker_fee_rate_override:
            return self.config.taker_fee_rate_override
        if not info.fees_enabled:
            return 0.0
        key = info.fee_type or "default"
        with self._lock:
            if key in self._fee_cache:
                return self._fee_cache[key]
        base_fee_bps = self._fetch_base_fee(info.token_ids[0] if info.token_ids else "")
        rate = float(
            resolve_taker_rate(
                fees_enabled=info.fees_enabled, fee_type=info.fee_type, base_fee_bps=base_fee_bps
            )
        )
        with self._lock:
            self._fee_cache[key] = rate
        return rate

    def _fetch_base_fee(self, token_id: str) -> int | None:
        if not token_id:
            return None
        try:
            payload = self.http.get(f"{self.config.clob_host}/fee-rate", {"token_id": token_id})
        except DataError:
            return None
        if isinstance(payload, dict):
            value = payload.get("base_fee")
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        return None

    # ------------------------------------------------------------- grouping
    def build_binary_group(self, info: MarketInfo, books: dict[str, OrderBook]) -> MarketGroup | None:
        """Build a two-outcome group, assigning Yes/No by outcome name."""
        if not info.is_binary:
            return None
        names = info.outcomes or ("Yes", "No")
        yes_idx = info.yes_index
        if yes_idx is None:
            # Non-Yes/No binary (for example a two-player match). Treat the
            # first outcome as the reference side; the pair is still exhaustive.
            yes_idx = 0
        no_idx = 1 - yes_idx
        outcomes = tuple(
            Outcome(index=i, name=names[i] if i < len(names) else f"Outcome{i}", token_id=info.token_ids[i], book=books.get(info.token_ids[i]))
            for i in (yes_idx, no_idx)
        )
        if any(o.book is None for o in outcomes):
            return None
        return MarketGroup(
            group_id=info.condition_id or info.market_id,
            title=info.question,
            outcomes=outcomes,
            volume_24h=info.volume_24h,
            liquidity=info.liquidity,
            seconds_to_end=info.seconds_to_end,
            neg_risk=info.neg_risk,
            is_binary=True,
            fees_enabled=info.fees_enabled,
            tick_size=info.tick_size,
            metadata={
                "market_id": info.market_id,
                "slug": info.slug,
                "fee_type": info.fee_type,
                "min_order_size": str(info.min_order_size),
                "yes_index": yes_idx,
                "event_id": info.event_id,
                "condition_id": info.condition_id,
                "leg_condition_ids": (info.condition_id, info.condition_id),
            },
        )

    def build_basket_group(self, infos: list[MarketInfo], books: dict[str, OrderBook]) -> MarketGroup | None:
        """Build an N-outcome neg-risk group from its constituent Yes markets."""
        if len(infos) < self.config.basket_min_outcomes:
            return None
        if len(infos) > self.config.basket_max_outcomes:
            return None
        outcomes: list[Outcome] = []
        for info in infos:
            yes_idx = info.yes_index
            if yes_idx is None:
                continue
            token_id = info.token_ids[yes_idx]
            book = books.get(token_id)
            if book is None:
                return None
            outcomes.append(
                Outcome(index=len(outcomes), name=info.group_item_title, token_id=token_id, book=book)
            )
        if len(outcomes) < self.config.basket_min_outcomes:
            return None
        first = infos[0]
        return MarketGroup(
            group_id=first.neg_risk_market_id or first.event_id or first.condition_id,
            title=first.event_title or first.question,
            outcomes=tuple(outcomes),
            volume_24h=max((i.volume_24h for i in infos), default=ZERO),
            liquidity=sum((i.liquidity for i in infos), ZERO),
            seconds_to_end=min(
                (i.seconds_to_end for i in infos if i.seconds_to_end is not None), default=None
            ),
            neg_risk=True,
            is_binary=False,
            fees_enabled=any(i.fees_enabled for i in infos),
            tick_size=first.tick_size,
            metadata={
                "event_id": first.event_id,
                "fee_type": first.fee_type,
                "n_markets": len(infos),
                "leg_condition_ids": tuple(
                    i.condition_id for i in infos if i.yes_index is not None
                ),
            },
        )

    # ------------------------------------------------------------- pipeline
    def scan(self) -> list[MarketGroup]:
        """Return all candidate groups with books attached.

        Every group that could produce a signal is returned; the per-cycle
        headcount cap is applied by the engine to its single-market pass, not
        here. Truncating to the top-N by volume here would drop the cross-market
        ladder candidates the pagination was added to find, because threshold
        ladders rank well below the top of the volume table.
        """
        self.stats.reset()
        groups: list[MarketGroup] = []

        binary_infos: list[MarketInfo] = []
        if self.config.arb_enabled or self.config.fade_enabled or self.config.cross_market_enabled:
            # The cross-market scan needs binary markets even when the
            # per-market strategies are off, and it needs to look deeper than
            # one page because ladder markets are not top-of-volume.
            pages = self.config.cross_market_scan_pages if self.config.cross_market_enabled else 1
            binary_infos = self.fetch_binary_markets(pages=pages)
            binary_infos = [i for i in binary_infos if i.is_binary and i.enable_order_book and i.accepting_orders]

        basket_infos: list[list[MarketInfo]] = []
        if self.config.basket_enabled:
            basket_infos = self.fetch_neg_risk_events()

        wanted: list[str] = []
        for info in binary_infos:
            wanted.extend(info.token_ids)
        for infos in basket_infos:
            for info in infos:
                if info.yes_index is not None:
                    wanted.append(info.token_ids[info.yes_index])

        books = self.fetch_books(wanted)

        for info in binary_infos:
            group = self.build_binary_group(info, books)
            if group is not None:
                groups.append(group)
        for infos in basket_infos:
            group = self.build_basket_group(infos, books)
            if group is not None:
                groups.append(group)

        self.stats.groups_built += len(groups)
        log.info(
            "scan complete: %d markets, %d books (%d missing), %d groups",
            self.stats.markets_seen,
            self.stats.books_fetched,
            self.stats.books_missing,
            len(groups),
        )
        return groups
