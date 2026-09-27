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
    # The churn guard is on by default: a stopped market is not re-bought.
    assert config.reentry_cooldown_seconds == 900
    assert config.fade_max_round_trip_ratio == pytest.approx(0.25)


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


def test_cross_market_scan_pages_must_be_positive():
    with pytest.raises(ConfigError):
        Config.from_env(cross_market_scan_pages=0)


def test_cross_market_defaults_are_off_and_ungated():
    config = Config()
    assert config.cross_market_enabled is False
    assert config.confirmed_cross_market_pairs == frozenset()


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


# ------------------------------------------------------------- pagination


class PagedHttp:
    """Serves ``total`` rows in 100-row pages, recording the offsets asked for."""

    def __init__(self, total: int):
        self.total = total
        self.offsets: list[int] = []

    def get(self, url, params=None):
        params = params or {}
        offset = params.get("offset", 0)
        self.offsets.append(offset)
        limit = params.get("limit", 100)
        rows = []
        for i in range(offset, min(offset + limit, self.total)):
            rows.append(dict(GAMMA_MARKET, id=str(i), conditionId=f"0x{i}", question=f"Q{i}"))
        return rows


def test_fetch_binary_markets_pages_past_the_100_row_api_cap():
    """Gamma returns at most 100 rows per request, so a deeper scan has to
    walk the offset rather than silently stopping at the first page."""
    scanner = MarketScanner(Config.from_env())
    scanner.http = PagedHttp(total=250)

    infos = scanner.fetch_binary_markets(limit=250, pages=3)

    assert len(infos) == 250
    assert scanner.http.offsets == [0, 100, 200]


def test_fetch_binary_markets_stops_early_on_a_short_page():
    scanner = MarketScanner(Config.from_env())
    scanner.http = PagedHttp(total=150)

    infos = scanner.fetch_binary_markets(limit=300, pages=5)

    assert len(infos) == 150
    # The third page came back short, so no fourth request is made.
    assert scanner.http.offsets == [0, 100]


def test_fetch_binary_markets_deduplicates_across_pages():
    """A market that shifts rank between page requests must not be counted
    twice, or the ladder scan would see phantom duplicates."""

    class ShiftingHttp(PagedHttp):
        def get(self, url, params=None):
            params = params or {}
            offset = params.get("offset", 0)
            self.offsets.append(offset)
            rows = []
            for i in range(offset, min(offset + 100, self.total)):
                rows.append(dict(GAMMA_MARKET, id=str(i), conditionId=f"0x{i}", question=f"Q{i}"))
            if offset == 0:
                rows.append(dict(GAMMA_MARKET, id="100", conditionId="0x100", question="dup"))
            return rows

    scanner = MarketScanner(Config.from_env())
    scanner.http = ShiftingHttp(total=150)

    infos = scanner.fetch_binary_markets(limit=200, pages=2)

    assert len({i.market_id for i in infos}) == len(infos)


def test_scanner_stats_describe_a_single_scan_not_a_running_total():
    """The "scan complete" counters must reset each scan.

    They used to accumulate forever, so the log line ("N markets") climbed
    while each cycle actually scanned the same capped number.
    """
    scanner = MarketScanner(Config.from_env())
    scanner.http = PagedHttp(total=250)

    scanner.fetch_binary_markets(limit=100, pages=1)
    first = scanner.stats.markets_seen
    assert first == 100

    scanner.stats.reset()
    scanner.fetch_binary_markets(limit=100, pages=1)
    assert scanner.stats.markets_seen == 100


def test_doctor_reports_data_failure_without_crashing(monkeypatch, capsys):
    """Regression: an unreachable Gamma must not crash `doctor`.

    `markets` was bound only inside the fetch try-block, so when the request
    raised DataError the next section read an unbound local and the whole
    command died with UnboundLocalError instead of printing FAIL.
    """
    from polymarket_bot import cli
    from polymarket_bot.data import DataError

    def boom(self, limit=None, pages=1):
        raise DataError("request failed after 3 attempts: gamma down")

    monkeypatch.setattr(MarketScanner, "fetch_binary_markets", boom)

    rc = cli.main(["doctor"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "FAIL: request failed" in out
    assert "doctor: FAIL" in out
    assert "SKIP: no market to inspect" in out

