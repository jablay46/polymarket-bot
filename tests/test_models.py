"""Tests for order book parsing and book-walking helpers.

The most important test here is that the best ask is the *lowest* price, since
the CLOB returns asks sorted high-to-low. Reading asks[0] as the best ask is
the bug this suite exists to prevent.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from polymarket_bot.models import OrderBook, quantize_down

# A book shaped the way the live CLOB actually returns one: asks descending,
# bids ascending, so element zero is the worst price on both sides.
RAW_CLOB_BOOK = {
    "asset_id": "abc",
    "asks": [
        {"price": "0.999", "size": "10211355.21"},
        {"price": "0.998", "size": "105410"},
        {"price": "0.010", "size": "500"},
        {"price": "0.006", "size": "138028.3"},
    ],
    "bids": [
        {"price": "0.001", "size": "267547.59"},
        {"price": "0.002", "size": "157102.65"},
        {"price": "0.004", "size": "2483.99"},
    ],
    "tick_size": "0.001",
    "min_order_size": "5",
    "neg_risk": False,
}


def test_best_ask_is_lowest_price_not_first_element():
    book = OrderBook.from_api(RAW_CLOB_BOOK)
    # asks[0] in the raw payload is 0.999; the true best ask is 0.006.
    assert book.best_ask == Decimal("0.006")
    assert book.asks[0].price == Decimal("0.006")
    assert book.asks[-1].price == Decimal("0.999")


def test_best_bid_is_highest_price_not_first_element():
    book = OrderBook.from_api(RAW_CLOB_BOOK)
    assert book.best_bid == Decimal("0.004")
    assert book.bids[0].price == Decimal("0.004")
    assert book.bids[-1].price == Decimal("0.001")


def test_spread_and_mid_use_true_top_of_book():
    book = OrderBook.from_api(RAW_CLOB_BOOK)
    assert book.spread == Decimal("0.002")
    assert book.mid == Decimal("0.005")


def test_tick_size_and_min_order_size_parsed():
    book = OrderBook.from_api(RAW_CLOB_BOOK)
    assert book.tick_size == Decimal("0.001")
    assert book.min_order_size == Decimal("5")
    assert book.neg_risk is False


def test_legacy_tuple_levels_are_supported():
    book = OrderBook.from_api({"asks": [["0.60", "10"], ["0.55", "5"]], "bids": [["0.50", "3"]]})
    assert book.best_ask == Decimal("0.55")
    assert book.best_bid == Decimal("0.50")


def test_zero_and_negative_levels_are_dropped():
    book = OrderBook.from_api(
        {"asks": [{"price": "0.50", "size": "10"}, {"price": "0", "size": "99"}], "bids": []}
    )
    assert len(book.asks) == 1
    assert book.best_bid is None


def test_empty_payload_raises():
    with pytest.raises(ValueError):
        OrderBook.from_api({})


def test_cost_to_buy_walks_levels():
    book = OrderBook.from_api(
        {"asks": [{"price": "0.60", "size": "10"}, {"price": "0.70", "size": "10"}], "bids": []}
    )
    fill = book.cost_to_buy(Decimal("15"))
    assert fill.shares == Decimal("15")
    assert fill.usd == Decimal("10") * Decimal("0.60") + Decimal("5") * Decimal("0.70")
    assert fill.worst_price == Decimal("0.70")
    assert fill.levels_used == 2
    assert fill.avg_price == fill.usd / Decimal("15")


def test_cost_to_buy_respects_price_limit():
    book = OrderBook.from_api(
        {"asks": [{"price": "0.60", "size": "10"}, {"price": "0.90", "size": "10"}], "bids": []}
    )
    fill = book.cost_to_buy(Decimal("15"), price_limit=Decimal("0.80"))
    assert fill.shares == Decimal("10")
    assert not fill.complete


def test_cost_to_buy_reports_incomplete_when_book_is_thin():
    book = OrderBook.from_api({"asks": [{"price": "0.50", "size": "4"}], "bids": []})
    fill = book.cost_to_buy(Decimal("10"))
    assert fill.shares == Decimal("4")
    assert not fill.complete  # only 4 of the 10 requested shares filled
    assert fill.requested == Decimal("10")


def test_shares_for_budget_spends_at_most_the_budget():
    book = OrderBook.from_api(
        {"asks": [{"price": "0.50", "size": "10"}, {"price": "0.60", "size": "10"}], "bids": []}
    )
    fill = book.shares_for_budget(Decimal("8"))
    assert fill.usd <= Decimal("8")
    assert fill.shares == Decimal("10") + (Decimal("8") - Decimal("5")) / Decimal("0.60")


def test_proceeds_to_sell_walks_bids():
    book = OrderBook.from_api(
        {"asks": [], "bids": [{"price": "0.45", "size": "5"}, {"price": "0.40", "size": "5"}]}
    )
    fill = book.proceeds_to_sell(Decimal("8"))
    assert fill.shares == Decimal("8")
    assert fill.usd == Decimal("5") * Decimal("0.45") + Decimal("3") * Decimal("0.40")


def test_has_two_sided_market():
    assert OrderBook.from_api(RAW_CLOB_BOOK).has_two_sided_market()
    assert not OrderBook.from_api({"asks": [{"price": "0.5", "size": "1"}], "bids": []}).has_two_sided_market()


def test_quantize_down_rounds_toward_zero():
    assert quantize_down(Decimal("9.99"), Decimal("1")) == Decimal("9")
    assert quantize_down(Decimal("0.999"), Decimal("0.01")) == Decimal("0.99")
    assert quantize_down(Decimal("5"), Decimal("1")) == Decimal("5")
