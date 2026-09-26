"""Cross-market logical relations: threshold ladders on the same subject.

Some Polymarket questions form a numeric ladder on a shared subject and
resolution window — for example "Will BTC exceed $90k by Dec 31?" and
"Will BTC exceed $100k by Dec 31?". Exceeding the higher threshold implies
exceeding every lower one, so the two markets are not independent: their
Yes probabilities must be non-increasing in the threshold,
``price(lower) >= price(higher)``. When the market violates that — a higher
threshold priced above a lower one — the same $1-set arbitrage that exists
between Yes and No inside one market exists across the two markets instead.

IMPORTANT — this module detects a *candidate* relation, not a proof. It
does not read the market's resolution rules, so it cannot tell "BTC exceeds
$90k by Dec 31 UTC, Binance spot" from "BTC exceeds $90k by Dec 31 UTC,
Coinbase spot" — two questions that look identical to this regex but are
not the same claim, and are not bound by the implication at all. A false
grouping does not average out the way a losing bet does: if the relation
does not actually hold, the "arbitrage" is a directional bet with a
misleading label. Every relation this module reports should be reviewed by
a human before it is traded.

That review is enforced, not merely requested. In paper mode
:class:`CrossMarketArbitrageStrategy` will price any detected pair so an
operator can watch it fill; in live mode it refuses every pair whose two
condition ids are not named in ``POLYMARKET_BOT_CROSS_MARKET_CONFIRMED_PAIRS``,
and that list starts empty. Detection can be automatic because the gate on
the way to real money is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from .models import ZERO

# Words that mark a threshold as crossed from above or from below. Kept as
# shared fragments so the threshold matcher and the direction matcher can
# never drift apart.
_ABOVE_WORDS = r"exceed|above|over|reach|hit|surpass|more than|greater than"
_BELOW_WORDS = r"no more than|not above|not more than|at most|below|under|less than|lower than"

# Matches "$90k", "$90,000", "90000", "$1.5m" style thresholds in a question.
# Below-markers come first so "no more than $90k" reads as one below-threshold
# rather than "more than $90k" (an above-threshold) found mid-phrase.
_THRESHOLD_RE = re.compile(
    rf"(?:{_BELOW_WORDS}|{_ABOVE_WORDS}|>=?|<=?)\s*\$?"
    r"(\d[\d,]*\.?\d*)\s*(k|m|thousand|million)?",
    re.IGNORECASE,
)

_MULTIPLIERS = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000}

_DIRECTION_RE = re.compile(rf"\b(?:{_BELOW_WORDS}|{_ABOVE_WORDS})\b|>=?|<=?", re.IGNORECASE)
_BELOW_ONLY_RE = re.compile(rf"^(?:{_BELOW_WORDS}|<=?)$", re.IGNORECASE)


def extract_threshold(question: str) -> Decimal | None:
    """Pull the first numeric threshold out of a question, or None.

    Handles both directions, so "below $90k" yields a threshold just as
    "above $90k" does. A market whose threshold cannot be read is skipped,
    which only costs one candidate pair.

    Deliberately takes only the *first* match: a question with two numbers
    ("up 10% to $90k") is ambiguous enough that guessing wrong is worse
    than skipping it.
    """
    match = _THRESHOLD_RE.search(question)
    if not match:
        return None
    try:
        number = Decimal(match.group(1).replace(",", ""))
    except Exception:  # noqa: BLE001 - malformed number, not a threshold
        return None
    suffix = (match.group(2) or "").lower()
    return number * Decimal(_MULTIPLIERS.get(suffix, 1))


def extract_direction(question: str) -> str:
    """Whether the question is an "above" or "below" threshold claim.

    Defaults to ``"above"`` when nothing matches, which is the common shape
    for these markets ("Will BTC exceed $100k?"). The two directions imply
    opposite ladders, so getting this wrong inverts the relation.
    """
    match = _DIRECTION_RE.search(question)
    if not match:
        return "above"
    return "below" if _BELOW_ONLY_RE.match(match.group(0).strip()) else "above"


def subject_key(question: str) -> str:
    """Group key for markets that plausibly share a subject.

    Strips the threshold and comparison words out of the question so that
    "Will BTC exceed $90k by Dec 31?" and "Will BTC exceed $100k by Dec 31?"
    collapse to the same key. Direction words are stripped too, so an "above"
    and a "below" market on one subject share a key and are separated by the
    direction field instead of by accident. This is deliberately crude — see
    the module docstring on why every group this produces needs human review
    before any of it trades.
    """
    text = _THRESHOLD_RE.sub(" ", question)
    text = _DIRECTION_RE.sub(" ", text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


@dataclass(frozen=True)
class ThresholdCandidate:
    """One market on a candidate threshold ladder."""

    market_id: str
    condition_id: str
    question: str
    threshold: Decimal
    yes_token_id: str
    no_token_id: str
    direction: str = "above"
    """Which way the threshold is crossed: ``"above"`` or ``"below"``.

    This is not cosmetic. A ladder only implies anything when every member
    crosses its threshold the same way. "BTC above $90k" implies "BTC above
    $100k" is *false* — it is the other way round — and a group that mixes
    the two has no relation at all, so pricing one as an arbitrage would be
    pricing a coin flip. ``group_threshold_ladders`` therefore refuses to
    mix directions, and ``cross_market`` refuses to price a pair whose
    directions disagree.
    """
    # Venue settings travel with the candidate because the two legs of a
    # cross-market pair sit in different markets and can disagree.
    tick_size: Decimal = Decimal("0.01")
    neg_risk: bool = False
    volume_24h: Decimal = ZERO
    liquidity: Decimal = ZERO
    seconds_to_end: float | None = None
    fees_enabled: bool = False
    fee_type: str | None = None


def group_threshold_ladders(
    candidates: list[ThresholdCandidate],
) -> dict[str, list[ThresholdCandidate]]:
    """Group candidates sharing a subject key, sorted low-to-high threshold.

    Only groups of 2+ are returned — a ladder of one has no pair to check.
    A group having 2+ members is a *candidate* for review, not a confirmed
    relation; see the module docstring.

    Grouping is by subject *and* direction. Mixing "above $90k" with "below
    $100k" in one ladder would put two markets on the same key whose prices
    are not ordered at all, and the pair check would read noise as edge.
    """
    groups: dict[str, list[ThresholdCandidate]] = {}
    for candidate in candidates:
        key = f"{subject_key(candidate.question)}|{candidate.direction}"
        groups.setdefault(key, []).append(candidate)
    return {
        key: sorted(members, key=lambda c: c.threshold)
        for key, members in groups.items()
        if len(members) >= 2
    }


def adjacent_pairs(
    ladder: list[ThresholdCandidate],
) -> list[tuple[ThresholdCandidate, ThresholdCandidate]]:
    """Consecutive (lower, higher) pairs in an already-sorted ladder.

    Only adjacent pairs are checked, not every combination: if $90k and
    $100k are both mispriced against $95k, trading the two adjacent pairs
    captures it; the non-adjacent $90k/$100k pair adds execution risk
    (two more legs) for no edge a human reviewing adjacent pairs would miss.
    """
    return list(zip(ladder, ladder[1:]))
