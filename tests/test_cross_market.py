"""Tests for threshold-ladder relation detection and cross-market pricing.

The pricing tests do not just check that a signal appears — they settle it.
Each one takes the legs the strategy chose, works out which tokens pay $1 in
every state the relation allows, and asserts the minimum payout covers the
cost. That invariant is the whole claim being made, and checking it directly
is what catches a pair built on the wrong side of the implication.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from polymarket_bot.config import Config
from polymarket_bot.cross_market import CrossMarketArbitrageStrategy, candidate_from_market
from polymarket_bot.fees import FeeModel
from polymarket_bot.models import ONE, ZERO, OrderBook
from polymarket_bot.relations import (
    ThresholdCandidate,
    adjacent_pairs,
    extract_direction,
    extract_threshold,
    group_threshold_ladders,
    subject_key,
)

FEE_FREE = FeeModel.for_market(fees_enabled=False)


def make_config(**overrides) -> Config:
    defaults = dict(
        cross_market_min_edge=0.02,
        cross_market_max_edge=0.15,
        arb_max_spread=0.05,
        cross_market_min_top_size=0.0,
        # Keep the default volume/liquidity floors out of the way so unit
        # tests can price a pair without staging market statistics.
        cross_market_min_volume_24h=0.0,
        cross_market_min_liquidity=0.0,
        cross_market_min_seconds_left=0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def book(asks, bids) -> OrderBook:
    return OrderBook.from_api(
        {
            "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
            "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
            "tick_size": "0.01",
        }
    )


def _candidate(market_id, question, threshold, direction="above") -> ThresholdCandidate:
    return ThresholdCandidate(
        market_id=market_id,
        condition_id=f"0x{market_id}",
        question=question,
        threshold=Decimal(threshold),
        yes_token_id=f"yes-{market_id}",
        no_token_id=f"no-{market_id}",
        direction=direction,
    )


def above_pair() -> tuple[ThresholdCandidate, ThresholdCandidate]:
    """A clean ascending "above" ladder: $90k (lower) and $100k (higher)."""
    return (
        _candidate("lo", "Will Bitcoin be above $90,000 by December 31?", "90000"),
        _candidate("hi", "Will Bitcoin be above $100,000 by December 31?", "100000"),
    )


def settle(signal, winners: set[str]) -> Decimal:
    """Cash one $1-set returns when `winners` are the tokens paying $1."""
    return sum((ONE for leg in signal.legs if leg.token_id in winners), ZERO)


# --------------------------------------------------------------- relations


def test_extract_threshold_reads_k_and_comma_suffixes():
    assert extract_threshold("Will BTC exceed $90,000 by Dec 31?") == Decimal("90000")
    assert extract_threshold("Will BTC exceed $90k by Dec 31?") == Decimal("90000")
    assert extract_threshold("Will BTC exceed $1.5m by Dec 31?") == Decimal("1500000")


def test_extract_threshold_returns_none_without_a_number():
    assert extract_threshold("Will it rain tomorrow?") is None


def test_extract_threshold_ignores_an_article_before_the_number():
    assert extract_threshold("Will BTC exceed the $90,000 mark by Dec 31?") == Decimal("90000")
    assert extract_threshold("Will BTC reach a $90k level?") == Decimal("90000")


def test_extract_threshold_does_not_read_the_letter_m_of_a_word_as_a_suffix():
    """Regression: "the $90,000 mark" parsed as 90 billion.

    The k/m suffix matcher must only accept a suffix that is a whole word, or
    the "m" of "mark" is read as "million".
    """
    assert extract_threshold("Will BTC exceed the $90,000 mark?") == Decimal("90000")


def test_extract_threshold_handles_at_least():
    assert extract_threshold("Will BTC be at least $90,000?") == Decimal("90000")


def test_extract_direction_distinguishes_above_and_below():
    assert extract_direction("Will BTC be above $90k?") == "above"
    assert extract_direction("Will BTC exceed $90k?") == "above"
    assert extract_direction("Will BTC be below $90k?") == "below"
    assert extract_direction("Will BTC be under $90k?") == "below"
    # Defaults to "above" when nothing marks a direction.
    assert extract_direction("Will BTC hit $90k?") == "above"


def test_extract_direction_reads_double_negatives_as_above():
    """Regression: "not less than $90k" means at or above, not below.

    The bare "less than" substring matched first and inverted the ladder.
    """
    assert extract_direction("Will BTC be not less than $90k?") == "above"
    assert extract_direction("Will BTC be no less than $90k?") == "above"
    assert extract_direction("Will BTC be no more than $90k?") == "below"


def test_subject_key_collapses_same_subject_different_thresholds():
    a = subject_key("Will Bitcoin exceed $90,000 by December 31?")
    b = subject_key("Will Bitcoin exceed $100k by December 31?")
    assert a == b


def test_subject_key_treats_above_and_below_wording_alike():
    """The direction lives in its own field, so the key itself must not split
    on it — otherwise the same subject would land in two unrelated buckets."""
    a = subject_key("Will Bitcoin be above $90,000 by December 31?")
    b = subject_key("Will Bitcoin be below $90,000 by December 31?")
    assert a == b


def test_subject_key_separates_different_subjects():
    a = subject_key("Will Bitcoin exceed $90,000 by December 31?")
    b = subject_key("Will Ethereum exceed $5,000 by December 31?")
    assert a != b


def test_subject_key_ignores_threshold_filler_words():
    """"the $90k mark" and "exceed $90k" are the same subject."""
    a = subject_key("Will Bitcoin exceed $90,000 by December 31?")
    b = subject_key("Will Bitcoin exceed the $90,000 mark by December 31?")
    assert a == b


def test_group_threshold_ladders_needs_at_least_two_members():
    single = [_candidate("m1", "Will Ethereum exceed $5,000 by December 31?", "5000")]
    assert group_threshold_ladders(single) == {}


def test_group_threshold_ladders_sorts_low_to_high():
    candidates = [
        _candidate("m1", "Will Bitcoin exceed $110k by December 31?", "110000"),
        _candidate("m2", "Will Bitcoin exceed $90,000 by December 31?", "90000"),
        _candidate("m3", "Will Bitcoin exceed $100k by December 31?", "100000"),
    ]
    groups = group_threshold_ladders(candidates)
    assert len(groups) == 1
    ladder = next(iter(groups.values()))
    assert [c.threshold for c in ladder] == [Decimal("90000"), Decimal("100000"), Decimal("110000")]


def test_group_threshold_ladders_never_mixes_directions():
    """An "above" and a "below" market on the same subject are not ordered
    against each other, so grouping them would price noise as edge."""
    candidates = [
        _candidate("m1", "Will Bitcoin be above $90,000 by December 31?", "90000", "above"),
        _candidate("m2", "Will Bitcoin be below $100,000 by December 31?", "100000", "below"),
    ]
    groups = group_threshold_ladders(candidates)
    assert groups == {}


def test_adjacent_pairs_skips_non_adjacent_combinations():
    candidates = [
        _candidate("m1", "Will Bitcoin exceed $90,000 by December 31?", "90000"),
        _candidate("m2", "Will Bitcoin exceed $100k by December 31?", "100000"),
        _candidate("m3", "Will Bitcoin exceed $110k by December 31?", "110000"),
    ]
    ladder = group_threshold_ladders(candidates)[
        f"{subject_key(candidates[0].question)}|above"
    ]
    pairs = adjacent_pairs(ladder)
    assert len(pairs) == 2
    assert (pairs[0][0].threshold, pairs[0][1].threshold) == (Decimal("90000"), Decimal("100000"))
    assert (pairs[1][0].threshold, pairs[1][1].threshold) == (Decimal("100000"), Decimal("110000"))


# ------------------------------------------------------------- pricing


def test_above_ladder_pays_at_least_one_in_every_state():
    """The claim under test: on an "above" ladder the relation makes
    ``above(high) => above(low)``, so holding Yes(low) + No(high) is covered.

    States, as BTC lands: above $100k, between, below $90k.
    """
    lower, higher = above_pair()
    # Yes(low) ask 0.45, No(high) ask 0.45 -> cost 0.90, a real violation.
    books = {
        lower.yes_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")]),
        higher.no_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")]),
    }
    signal = CrossMarketArbitrageStrategy(make_config()).evaluate_pair(
        lower, higher, books, FEE_FREE, FEE_FREE
    )

    assert signal is not None
    assert signal.kind == "cross_market_arbitrage"
    assert signal.metadata["unverified_relation"] is True
    assert signal.metadata["direction"] == "above"
    # The safe side: Yes on the implied market (lower), No on the implying one.
    assert signal.legs[0].token_id == lower.yes_token_id
    assert signal.legs[1].token_id == higher.no_token_id

    cost = signal.cost_per_set
    assert cost == Decimal("0.90")
    for state, winners in [
        ("BTC above $100k", {lower.yes_token_id, higher.yes_token_id}),
        ("$90k-$100k", {lower.yes_token_id, higher.no_token_id}),
        ("BTC below $90k", {lower.no_token_id, higher.no_token_id}),
    ]:
        payout = settle(signal, winners)
        assert payout >= ONE, f"{state} pays {payout} for cost {cost}"


def test_below_ladder_uses_the_opposite_pair_and_still_pays_at_least_one():
    """On a "below" ladder the implication runs the other way, so the safe
    side is Yes(high) + No(low) — the mirror image of the above case."""
    lower = _candidate("lo", "Will Bitcoin be below $90,000 by December 31?", "90000", "below")
    higher = _candidate("hi", "Will Bitcoin be below $100,000 by December 31?", "100000", "below")
    books = {
        higher.yes_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")]),
        lower.no_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")]),
    }
    signal = CrossMarketArbitrageStrategy(make_config()).evaluate_pair(
        lower, higher, books, FEE_FREE, FEE_FREE
    )

    assert signal is not None
    assert signal.metadata["direction"] == "below"
    # Mirrored: Yes on the implied market (higher), No on the implying one.
    assert signal.legs[0].token_id == higher.yes_token_id
    assert signal.legs[1].token_id == lower.no_token_id

    cost = signal.cost_per_set
    for state, winners in [
        ("BTC below $90k", {lower.yes_token_id, higher.yes_token_id}),
        ("$90k-$100k", {lower.no_token_id, higher.yes_token_id}),
        ("BTC above $100k", {lower.no_token_id, higher.no_token_id}),
    ]:
        payout = settle(signal, winners)
        assert payout >= ONE, f"{state} pays {payout} for cost {cost}"


def test_mixed_direction_pair_is_refused():
    lower = _candidate("lo", "Will Bitcoin be above $90,000 by December 31?", "90000", "above")
    higher = _candidate("hi", "Will Bitcoin be below $100,000 by December 31?", "100000", "below")
    books = {
        lower.yes_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")]),
        higher.no_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")]),
    }
    signal = CrossMarketArbitrageStrategy(make_config()).evaluate_pair(
        lower, higher, books, FEE_FREE, FEE_FREE
    )
    assert signal is None


def test_cross_market_finds_nothing_when_consistently_priced():
    """Correct ordering (P(low) >= P(high)) leaves the safe side at or above
    $1, so there is nothing to buy."""
    lower, higher = above_pair()
    # Yes(low) 0.58 and No(high) 0.50 -> 1.08, over a dollar.
    books = {
        lower.yes_token_id: book(asks=[("0.58", "500")], bids=[("0.57", "500")]),
        higher.no_token_id: book(asks=[("0.50", "500")], bids=[("0.49", "500")]),
    }
    signal = CrossMarketArbitrageStrategy(make_config()).evaluate_pair(
        lower, higher, books, FEE_FREE, FEE_FREE
    )
    assert signal is None


def test_cross_market_respects_min_edge():
    lower, higher = above_pair()
    # Cost 0.99 -> 1% edge, under the 2% default minimum.
    books = {
        lower.yes_token_id: book(asks=[("0.50", "500")], bids=[("0.49", "500")]),
        higher.no_token_id: book(asks=[("0.49", "500")], bids=[("0.48", "500")]),
    }
    signal = CrossMarketArbitrageStrategy(make_config()).evaluate_pair(
        lower, higher, books, FEE_FREE, FEE_FREE
    )
    assert signal is None


def test_cross_market_rejects_thin_top_of_book():
    lower, higher = above_pair()
    books = {
        lower.yes_token_id: book(asks=[("0.45", "5")], bids=[("0.44", "500")]),
        higher.no_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")]),
    }
    signal = CrossMarketArbitrageStrategy(make_config(cross_market_min_top_size=100)).evaluate_pair(
        lower, higher, books, FEE_FREE, FEE_FREE
    )
    assert signal is None


def test_cross_market_fees_reduce_the_edge():
    lower, higher = above_pair()
    books = {
        lower.yes_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")]),
        higher.no_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")]),
    }
    strategy = CrossMarketArbitrageStrategy(make_config())
    fee_5 = FeeModel.for_market(fees_enabled=True, fee_type="crypto_fees_v2")
    gross = strategy.evaluate_pair(lower, higher, books, FEE_FREE, FEE_FREE)
    net = strategy.evaluate_pair(lower, higher, books, fee_5, fee_5)
    assert gross is not None and net is not None
    assert net.edge_per_set < gross.edge_per_set


def test_signal_stamps_its_own_per_set_fee():
    """Risk sizing must use the fees the pair was priced with. The two legs can
    sit in markets with different fee types, so the single group fee the engine
    passes later is not a safe substitute."""
    lower, higher = above_pair()
    books = {
        lower.yes_token_id: book(asks=[("0.40", "500")], bids=[("0.39", "500")]),
        higher.no_token_id: book(asks=[("0.40", "500")], bids=[("0.39", "500")]),
    }
    strategy = CrossMarketArbitrageStrategy(make_config(cross_market_max_edge=0.5))
    fee = FeeModel.for_market(fees_enabled=True, fee_type="crypto_fees_v2")
    signal = strategy.evaluate_pair(lower, higher, books, fee, fee)
    assert signal is not None
    stamped = Decimal(signal.metadata["fee_per_set"])
    assert stamped > ZERO
    # The stamped fee must be the one that produced the reported net edge.
    gross = Decimal(signal.metadata["gross_edge"])
    assert gross - stamped == signal.edge_per_set


def test_cross_market_needs_two_sided_books():
    lower, higher = above_pair()
    books = {
        lower.yes_token_id: book(asks=[("0.45", "500")], bids=[]),
        higher.no_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")]),
    }
    signal = CrossMarketArbitrageStrategy(make_config()).evaluate_pair(
        lower, higher, books, FEE_FREE, FEE_FREE
    )
    assert signal is None


def test_cross_market_missing_book_is_refused():
    lower, higher = above_pair()
    books = {lower.yes_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")])}
    signal = CrossMarketArbitrageStrategy(make_config()).evaluate_pair(
        lower, higher, books, FEE_FREE, FEE_FREE
    )
    assert signal is None


# ------------------------------------------------------------- candidates


class _Market:
    def __init__(self, market_id, question, token_ids, **kwargs):
        self.market_id = market_id
        self.condition_id = f"0x{market_id}"
        self.question = question
        self.token_ids = token_ids
        self.outcomes = kwargs.get("outcomes", ("Yes", "No"))
        if "yes_index" in kwargs:
            self.yes_index = kwargs["yes_index"]
        else:
            self.yes_index = next(
                (i for i, n in enumerate(self.outcomes) if str(n).strip().lower() == "yes"),
                None,
            )
        self.tick_size = kwargs.get("tick_size", Decimal("0.01"))
        self.neg_risk = kwargs.get("neg_risk", False)
        self.volume_24h = kwargs.get("volume_24h", 50000)
        self.liquidity = kwargs.get("liquidity", 20000)
        self.seconds_to_end = kwargs.get("seconds_to_end", 86400)
        self.fees_enabled = kwargs.get("fees_enabled", False)
        self.fee_type = kwargs.get("fee_type", None)


def test_candidate_from_market_reads_threshold_and_direction():
    market = _Market("m1", "Will Bitcoin be below $90,000 by December 31?", ("yes", "no"))
    candidate = candidate_from_market(market)
    assert candidate is not None
    assert candidate.threshold == Decimal("90000")
    assert candidate.direction == "below"
    assert candidate.yes_token_id == "yes"
    assert candidate.no_token_id == "no"


def test_candidate_from_market_carries_venue_settings():
    """The two legs of a cross-market pair sit in different markets, so tick
    size and neg-risk have to travel with the candidate."""
    market = _Market(
        "m1",
        "Will Bitcoin be above $90,000 by December 31?",
        ("yes", "no"),
        tick_size=Decimal("0.001"),
        neg_risk=True,
    )
    candidate = candidate_from_market(market)
    assert candidate is not None
    assert candidate.tick_size == Decimal("0.001")
    assert candidate.neg_risk is True


def test_candidate_from_market_skips_non_binary_and_thresholdless():
    assert candidate_from_market(_Market("m1", "Who wins?", ("a", "b", "c"))) is None
    assert candidate_from_market(_Market("m2", "Will it rain?", ("yes", "no"))) is None


def test_candidate_from_market_resolves_yes_by_name_not_position():
    """Gamma does not guarantee Yes is first in clobTokenIds.

    Taking token_ids[0] as Yes would silently swap the two legs of the pair,
    pricing a directional bet as an arbitrage. The candidate must follow the
    outcome name instead.
    """
    market = _Market(
        "m1",
        "Will Bitcoin be above $90,000 by December 31?",
        ("tok-no", "tok-yes"),
        outcomes=("No", "Yes"),
    )
    candidate = candidate_from_market(market)
    assert candidate is not None
    assert candidate.yes_token_id == "tok-yes"
    assert candidate.no_token_id == "tok-no"


def test_candidate_from_market_uses_market_yes_index_when_present():
    market = _Market(
        "m1",
        "Will Bitcoin be above $90,000 by December 31?",
        ("tok-a", "tok-b"),
        outcomes=("Foo", "Bar"),
        yes_index=1,
    )
    candidate = candidate_from_market(market)
    assert candidate is not None
    assert candidate.yes_token_id == "tok-b"
    assert candidate.no_token_id == "tok-a"


def test_candidate_from_market_falls_back_to_first_token_without_labels():
    """A two-outcome market with no Yes/No labels still partitions $1."""
    market = _Market("m1", "Will Bitcoin be above $90,000 by December 31?", ("a", "b"))
    candidate = candidate_from_market(market)
    assert candidate is not None
    assert candidate.yes_token_id == "a"
    assert candidate.no_token_id == "b"


# --------------------------------------------------------------- live gating


def _violation_books(lower, higher):
    return {
        lower.yes_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")]),
        higher.no_token_id: book(asks=[("0.45", "500")], bids=[("0.44", "500")]),
    }


def test_live_mode_refuses_a_pair_that_is_not_confirmed():
    """The relation is a guess from wording. In live mode an unconfirmed pair
    must not reach execution, because a wrong guess is a naked directional bet."""
    lower, higher = above_pair()
    strategy = CrossMarketArbitrageStrategy(make_config(mode="live"))
    signal = strategy.evaluate_pair(
        lower, higher, _violation_books(lower, higher), FEE_FREE, FEE_FREE
    )
    assert signal is None


def test_paper_mode_prices_an_unconfirmed_pair_but_flags_it():
    lower, higher = above_pair()
    signal = CrossMarketArbitrageStrategy(make_config(mode="paper")).evaluate_pair(
        lower, higher, _violation_books(lower, higher), FEE_FREE, FEE_FREE
    )
    assert signal is not None
    assert signal.metadata["confirmed_pair"] is False
    assert signal.metadata["unverified_relation"] is True


def test_live_mode_accepts_a_confirmed_pair():
    lower, higher = above_pair()
    pair_id = f"{lower.condition_id}:{higher.condition_id}"
    strategy = CrossMarketArbitrageStrategy(
        make_config(mode="live", cross_market_confirmed_pairs=pair_id)
    )
    signal = strategy.evaluate_pair(
        lower, higher, _violation_books(lower, higher), FEE_FREE, FEE_FREE
    )
    assert signal is not None
    assert signal.metadata["confirmed_pair"] is True


def test_confirmed_pairs_accept_either_order_and_separators():
    """An operator pastes the ids in whatever order a log printed them, so the
    lookup has to be order-insensitive."""
    config = make_config(
        cross_market_confirmed_pairs="0xa:0xb; 0xd:0xc,0xb:0xa"
    )
    assert ("0xa", "0xb") in config.confirmed_cross_market_pairs
    assert ("0xc", "0xd") in config.confirmed_cross_market_pairs
    assert len(config.confirmed_cross_market_pairs) == 2


def test_confirmed_pairs_ignores_malformed_entries():
    config = make_config(cross_market_confirmed_pairs="0xa,0xb:0xc:0xd,:,0xe:0xf")
    assert config.confirmed_cross_market_pairs == frozenset({("0xe", "0xf")})


def test_live_gating_precedes_pricing_so_a_confirmed_pair_still_needs_an_edge():
    """Confirmation only unlocks the pair; it does not manufacture an edge."""
    lower, higher = above_pair()
    pair_id = f"{lower.condition_id}:{higher.condition_id}"
    strategy = CrossMarketArbitrageStrategy(
        make_config(mode="live", cross_market_confirmed_pairs=pair_id)
    )
    # Priced fairly: Yes(low) 0.58, No(high) 0.42 -> cost 1.00, no edge.
    books = {
        lower.yes_token_id: book(asks=[("0.58", "500")], bids=[("0.57", "500")]),
        higher.no_token_id: book(asks=[("0.42", "500")], bids=[("0.41", "500")]),
    }
    assert strategy.evaluate_pair(lower, higher, books, FEE_FREE, FEE_FREE) is None


def test_market_filters_reject_a_ladder_leg_in_a_dead_market():
    """A leg in a market with no volume cannot be exited, so the pair is
    refused even though the books price an edge."""
    lower, higher = above_pair()
    strategy = CrossMarketArbitrageStrategy(
        make_config(
            cross_market_min_volume_24h=25000.0,
            cross_market_min_liquidity=10000.0,
        )
    )
    dead = _candidate("lo", "Will Bitcoin be above $90,000 by December 31?", "90000")
    dead = replace(dead, volume_24h=Decimal("100"), liquidity=Decimal("100"))
    signal = strategy.evaluate_pair(
        dead, higher, _violation_books(dead, higher), FEE_FREE, FEE_FREE
    )
    assert signal is None


def test_signal_legs_carry_their_own_market_venue_settings():
    """Each leg keeps its own market's condition id and tick size, so the
    execution path does not have to guess which market a leg belongs to."""
    lower = replace(
        _candidate("lo", "Will Bitcoin be above $90,000 by December 31?", "90000"),
        tick_size=Decimal("0.001"),
        neg_risk=True,
    )
    higher = replace(
        _candidate("hi", "Will Bitcoin be above $100,000 by December 31?", "100000"),
        tick_size=Decimal("0.01"),
        neg_risk=False,
    )
    signal = CrossMarketArbitrageStrategy(make_config()).evaluate_pair(
        lower, higher, _violation_books(lower, higher), FEE_FREE, FEE_FREE
    )
    assert signal is not None
    by_token = {leg.token_id: leg for leg in signal.legs}
    assert by_token[lower.yes_token_id].condition_id == lower.condition_id
    assert by_token[lower.yes_token_id].tick_size == Decimal("0.001")
    assert by_token[lower.yes_token_id].neg_risk is True
    assert by_token[higher.no_token_id].condition_id == higher.condition_id
    assert by_token[higher.no_token_id].tick_size == Decimal("0.01")
    assert by_token[higher.no_token_id].neg_risk is False
    # Execution needs the per-leg condition ids to book one position across
    # two markets.
    assert set(signal.metadata["leg_condition_ids"]) == {
        lower.condition_id,
        higher.condition_id,
    }
    assert signal.metadata["cross_market"] is True
