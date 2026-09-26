"""Tests for the fee model.

Reference values come from the published Polymarket fee tables, where the
maximum taker fee is $1.25 per 100 shares at a price of 0.50 for the 0.05
rate categories.
"""

from __future__ import annotations

from decimal import Decimal

from polymarket_bot.fees import (
    DEFAULT_TAKER_RATE,
    FeeModel,
    category_from_fee_type,
    resolve_taker_rate,
    taker_fee,
)


def test_fee_type_maps_to_category():
    assert category_from_fee_type("sports_fees_v3") == "sports"
    assert category_from_fee_type("sports_fees_v2") == "sports"
    assert category_from_fee_type("crypto_fees_v2") == "crypto"
    assert category_from_fee_type("finance_prices_fees") == "finance"
    assert category_from_fee_type("politics_fees") == "politics"
    assert category_from_fee_type("zero_fees") == "geopolitics"
    assert category_from_fee_type(None) is None
    assert category_from_fee_type("something_unknown") is None


def test_fee_free_market_has_zero_rate_even_with_a_category():
    assert resolve_taker_rate(fees_enabled=False, fee_type="crypto_fees_v2") == Decimal("0")


def test_category_rates_are_applied():
    assert resolve_taker_rate(fees_enabled=True, fee_type="crypto_fees_v2") == Decimal("0.07")
    assert resolve_taker_rate(fees_enabled=True, fee_type="politics_fees") == Decimal("0.04")
    assert resolve_taker_rate(fees_enabled=True, fee_type="sports_fees_v3") == Decimal("0.05")


def test_override_wins():
    assert resolve_taker_rate(fees_enabled=False, override=0.03) == Decimal("0.03")
    assert resolve_taker_rate(fees_enabled=True, fee_type="crypto_fees_v2", override=0.01) == Decimal("0.01")


def test_base_fee_bps_falls_back_to_default_rate():
    # The live endpoint reports 1000 bps, which is not a real per-trade rate.
    # We must not treat it as 10%.
    assert resolve_taker_rate(fees_enabled=True, base_fee_bps=1000) == DEFAULT_TAKER_RATE
    assert resolve_taker_rate(fees_enabled=True, base_fee_bps=0) == Decimal("0")


def test_fee_peaks_at_half_price_matching_published_table():
    # 100 shares at 0.50 with a 0.05 rate is $1.25, the documented maximum.
    fee = taker_fee(Decimal("100"), Decimal("0.50"), Decimal("0.05"))
    assert fee == Decimal("1.25")
    # Symmetric around 0.50.
    assert taker_fee(Decimal("100"), Decimal("0.30"), Decimal("0.05")) == taker_fee(
        Decimal("100"), Decimal("0.70"), Decimal("0.05")
    )
    # Cheaper near the extremes.
    assert taker_fee(Decimal("100"), Decimal("0.90"), Decimal("0.05")) == Decimal("0.45")


def test_zero_inputs_produce_zero_fee():
    assert taker_fee(Decimal("0"), Decimal("0.5"), Decimal("0.05")) == Decimal("0")
    assert taker_fee(Decimal("10"), Decimal("0"), Decimal("0.05")) == Decimal("0")
    assert taker_fee(Decimal("10"), Decimal("0.5"), Decimal("0")) == Decimal("0")


def test_fee_model_for_market_wrapper():
    model = FeeModel.for_market(fees_enabled=True, fee_type="crypto_fees_v2")
    assert model.taker_rate == Decimal("0.07")
    assert not model.is_fee_free
    assert FeeModel.for_market(fees_enabled=False).is_fee_free


def test_maker_mode_pays_no_fee():
    model = FeeModel.for_market(fees_enabled=True, fee_type="crypto_fees_v2", maker=True)
    assert model.taker_rate == Decimal("0")
    assert model.buy_fee(Decimal("100"), Decimal("0.5")) == Decimal("0")


def test_safety_multiplier_scales_the_fee():
    model = FeeModel.for_market(fees_enabled=True, fee_type="crypto_fees_v2", safety_multiplier=2)
    assert model.buy_fee(Decimal("100"), Decimal("0.50")) == Decimal("3.5")  # 1.75 * 2


def test_marginal_rate_decreases_as_price_rises():
    model = FeeModel.for_market(fees_enabled=True, fee_type="crypto_fees_v2")
    assert model.marginal_rate_at(Decimal("0.50")) > model.marginal_rate_at(Decimal("0.90"))
