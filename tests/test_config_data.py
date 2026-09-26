"""Tests for configuration loading, .env handling, and data parsing."""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

from polymarket_bot.config import Config, ConfigError
from polymarket_bot.data import MarketScanner, parse_market
from polymarket_bot.env import load_dotenv


def test_env_file_values_reach_config(tmp_path, monkeypatch):
    """Regression test: .env must take effect, not be shadowed by defaults."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment line",
                "POLYMARKET_BOT_MODE=live",
                "POLYMARKET_BOT_ARB_MIN_EDGE=0.099",
                "POLYMARKET_BOT_KILL_SWITCH=true",
                "export POLYMARKET_BOT_MAX_ORDER_USD=42",
                'POLYMARKET_BOT_LOG_LEVEL="DEBUG"',
            ]
        )
    )
    for key in (
        "POLYMARKET_BOT_MODE",
        "POLYMARKET_BOT_ARB_MIN_EDGE",
        "POLYMARKET_BOT_KILL_SWITCH",
        "POLYMARKET_BOT_MAX_ORDER_USD",
        "POLYMARKET_BOT_LOG_LEVEL",
    ):
        monkeypatch.delenv(key, raising=False)

    load_dotenv(env_file, override=True)
    config = Config.from_env()

    assert config.mode == "live"
    assert config.arb_min_edge == pytest.approx(0.099)
    assert config.kill_switch is True
    assert config.max_order_usd == pytest.approx(42.0)
    assert config.log_level == "DEBUG"


def test_inline_comment_is_stripped(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("POLYMARKET_BOT_ARB_MIN_EDGE=0.05 # five percent\n")
    monkeypatch.delenv("POLYMARKET_BOT_ARB_MIN_EDGE", raising=False)
    load_dotenv(env_file, override=True)
    assert Config.from_env().arb_min_edge == pytest.approx(0.05)


def test_defaults_apply_without_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("POLYMARKET_BOT_") or key.startswith("POLYMARKET_"):
            monkeypatch.delenv(key, raising=False)
    config = Config.from_env()
    assert config.mode == "paper"
    assert config.arb_enabled is True
    assert config.basket_enabled is True
    assert config.fade_enabled is False
    assert config.kill_switch is False


def test_overrides_take_precedence(monkeypatch):
    monkeypatch.setenv("POLYMARKET_BOT_MAX_ORDER_USD", "10")
    config = Config.from_env(max_order_usd=999.0)
    assert config.max_order_usd == pytest.approx(999.0)


def test_invalid_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("POLYMARKET_BOT_MODE", "yolo")
    with pytest.raises(ConfigError):
        Config.from_env()


def test_inconsistent_edge_bounds_are_rejected():
    with pytest.raises(ConfigError):
        Config.from_env(arb_min_edge=0.2, arb_max_edge=0.1)


def test_out_of_range_values_are_rejected():
    with pytest.raises(ConfigError):
        Config.from_env(arb_min_edge=1.5)


def test_describe_redacts_secrets():
    config = Config.from_env(private_key="0xdeadbeef", clob_api_secret="shh")
    described = config.describe()
    assert described["private_key"] == "<set>"
    assert described["clob_api_secret"] == "<set>"
    assert "deadbeef" not in str(described)


# ------------------------------------------------------------ data parsing


GAMMA_MARKET = {
    "id": "123",
    "conditionId": "0xabc",
    "question": "Will it rain tomorrow?",
    "slug": "will-it-rain",
    "clobTokenIds": '["token-yes", "token-no"]',
    "outcomes": '["Yes", "No"]',
    "volume24hr": "50000",
    "liquidity": "25000",
    "endDate": "2027-01-01T00:00:00Z",
    "negRisk": False,
    "feesEnabled": True,
    "feeType": "politics_fees",
    "orderPriceMinTickSize": 0.01,
    "orderMinSize": 5,
}


def test_parse_market_extracts_fields():
    info = parse_market(GAMMA_MARKET)
    assert info is not None
    assert info.question == "Will it rain tomorrow?"
    assert info.token_ids == ("token-yes", "token-no")
    assert info.outcomes == ("Yes", "No")
    assert info.volume_24h == Decimal("50000")
    assert info.neg_risk is False
    assert info.fees_enabled is True
    assert info.fee_type == "politics_fees"
    assert info.is_binary is True
    assert info.yes_index == 0


def test_parse_market_identifies_yes_by_name_not_position():
    swapped = dict(GAMMA_MARKET, outcomes='["No", "Yes"]')
    info = parse_market(swapped)
    assert info is not None
    assert info.yes_index == 1


def test_parse_market_handles_list_fields():
    as_lists = dict(GAMMA_MARKET, clobTokenIds=["a", "b"], outcomes=["Yes", "No"])
    info = parse_market(as_lists)
    assert info is not None
    assert info.token_ids == ("a", "b")


def test_parse_market_returns_none_without_tokens():
    assert parse_market({"question": "no tokens here"}) is None
    assert parse_market({}) is None


def test_parse_market_marks_multi_outcome_as_non_binary():
    multi = dict(GAMMA_MARKET, clobTokenIds='["a","b","c"]', outcomes='["A","B","C"]')
    info = parse_market(multi)
    assert info is not None
    assert info.is_binary is False


def test_binary_group_pairs_the_two_outcomes():
    """Token ids and outcome names are parallel arrays; pairing must respect that."""
    # outcomes[0]="No" maps to clobTokenIds[0], so both arrays are swapped
    # together. The group must keep each name attached to its own token.
    info = parse_market(
        dict(GAMMA_MARKET, outcomes='["No", "Yes"]', clobTokenIds='["token-no", "token-yes"]')
    )
    scanner = MarketScanner(Config.from_env())
    from polymarket_bot.models import OrderBook

    books = {
        "token-yes": OrderBook.from_api(
            {"asks": [{"price": "0.40", "size": "100"}], "bids": [{"price": "0.39", "size": "100"}]}
        ),
        "token-no": OrderBook.from_api(
            {"asks": [{"price": "0.60", "size": "100"}], "bids": [{"price": "0.59", "size": "100"}]}
        ),
    }
    group = scanner.build_binary_group(info, books)
    assert group is not None
    by_name = {o.name: o for o in group.outcomes}
    assert set(by_name) == {"Yes", "No"}
    assert by_name["Yes"].token_id == "token-yes"
    assert by_name["No"].token_id == "token-no"
    assert by_name["Yes"].best_ask == Decimal("0.40")
    assert group.ask_sum() == Decimal("1.00")


def test_fee_rate_is_zero_when_fees_disabled():
    scanner = MarketScanner(Config.from_env())
    info = parse_market(dict(GAMMA_MARKET, feesEnabled=False))
    assert scanner.fee_rate_for(info) == 0.0


def test_fee_rate_uses_category_table():
    scanner = MarketScanner(Config.from_env())
    info = parse_market(dict(GAMMA_MARKET, feeType="crypto_fees_v2", feesEnabled=True))
    assert scanner.fee_rate_for(info) == pytest.approx(0.07)
