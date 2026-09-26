"""Broker tests.

The paper broker is fully exercised here. The live adapters are checked with a
*contract* test: if an official SDK is installed, every keyword argument the
adapter passes is bound against the real signature. This catches SDK drift
(renamed argument, renamed class) without needing credentials or network.
"""

from __future__ import annotations

import inspect
from decimal import Decimal

import pytest

from polymarket_bot.brokers import LegResult, LiveBroker, PaperBroker, _as_dict, build_broker
from polymarket_bot.config import Config

# --------------------------------------------------------------- paper broker


def make_config(**overrides) -> Config:
    return Config.from_env(**overrides)


def test_build_broker_returns_paper_broker_by_default():
    broker = build_broker(make_config(mode="paper"))
    assert isinstance(broker, PaperBroker)
    assert broker.name == "paper"
    assert broker.supports_balance is False


def test_paper_broker_buy_fills_the_full_notional():
    broker = PaperBroker(make_config())
    result = broker.buy(token_id="yes", usd=Decimal("50"), price=Decimal("0.25"))
    assert result.ok
    assert result.filled_usd == Decimal("50")
    assert result.filled_shares == Decimal("200")
    assert result.status == "matched"


def test_paper_broker_sell_returns_the_notional():
    broker = PaperBroker(make_config())
    result = broker.sell(token_id="yes", shares=Decimal("200"), price=Decimal("0.25"))
    assert result.ok
    assert result.filled_usd == Decimal("50")
    assert result.filled_shares == Decimal("200")


def test_paper_broker_rejects_invalid_orders():
    broker = PaperBroker(make_config())
    assert not broker.buy(token_id="yes", usd=Decimal("0"), price=Decimal("0.5")).ok
    assert not broker.buy(token_id="yes", usd=Decimal("10"), price=Decimal("0")).ok
    assert not broker.sell(token_id="yes", shares=Decimal("0"), price=Decimal("0.5")).ok


def test_paper_broker_holds_no_cash_of_its_own():
    """The portfolio owns paper cash, so the broker must not report a balance."""
    broker = PaperBroker(make_config())
    broker.buy(token_id="yes", usd=Decimal("100"), price=Decimal("0.5"))
    assert broker.balance_usd() == Decimal("0")


def test_paper_broker_records_orders_for_inspection():
    broker = PaperBroker(make_config())
    broker.buy(token_id="yes", usd=Decimal("10"), price=Decimal("0.5"))
    broker.sell(token_id="yes", shares=Decimal("20"), price=Decimal("0.5"))
    assert [o["side"] for o in broker.orders] == ["BUY", "SELL"]
    assert broker.cancel("paper-1") is True


# ------------------------------------------------------------ live broker gates


def test_live_broker_requires_a_private_key():
    from polymarket_bot.brokers import BrokerError

    with pytest.raises(BrokerError, match="PRIVATE_KEY"):
        LiveBroker(make_config(mode="live", private_key=""))


def test_live_broker_reports_a_clear_error_when_no_client_is_installed(monkeypatch):
    from polymarket_bot.brokers import BrokerError

    broker = LiveBroker.__new__(LiveBroker)
    broker.config = make_config(mode="live", private_key="0xdeadbeef")

    def explode():
        raise ImportError("no module named py_clob_client_v2")

    monkeypatch.setattr(broker, "_connect_sdk", explode)
    monkeypatch.setattr(broker, "_connect_v2", explode)
    with pytest.raises(BrokerError, match="no usable CLOB client"):
        broker._connect()


# ------------------------------------------------------------------- helpers


def test_as_dict_passes_dicts_through_and_extracts_model_dumps():
    assert _as_dict({"a": 1}) == {"a": 1}

    class Model:
        def model_dump(self):
            return {"ok": True}

    assert _as_dict(Model()) == {"ok": True}
    assert _as_dict(object()) == {}


def test_interpret_reads_an_accepted_order_shape():
    class Accepted:
        ok = True
        order_id = "abc"
        status = "matched"
        making_amount = 100
        taking_amount = 50

    result = LiveBroker._interpret(Accepted(), "tok", "BUY")
    assert result.ok
    assert result.order_id == "abc"
    assert result.filled_usd == Decimal("100")
    assert result.filled_shares == Decimal("50")


def test_interpret_reads_a_rejected_order_shape():
    class Rejected:
        ok = False
        code = "400"
        message = "not enough balance"

    result = LiveBroker._interpret(Rejected(), "tok", "BUY")
    assert not result.ok
    assert "400" in result.error and "not enough balance" in result.error


def test_interpret_reads_a_legacy_dict_response():
    result = LiveBroker._interpret(
        {"success": True, "orderID": "x", "status": "live", "makingAmount": "20", "takingAmount": "10"},
        "tok",
        "SELL",
    )
    assert result.ok
    assert result.filled_shares == Decimal("20")
    assert result.filled_usd == Decimal("10")


# ------------------------------------------------- live SDK contract (optional)

_SKIP = "official Polymarket client not installed"


def _bind(fn, **kwargs):
    """Bind kwargs against a signature, ignoring the bound `self`."""
    signature = inspect.signature(fn)
    params = list(signature.parameters.values())
    if params and params[0].name in {"self", "cls"}:
        signature = signature.replace(parameters=params[1:])
    signature.bind(**kwargs)


def test_py_clob_client_v2_adapter_matches_the_installed_sdk():
    """Every kwarg our v2 adapter sends must exist in the installed client."""
    v2 = pytest.importorskip("py_clob_client_v2", reason=_SKIP)

    _bind(v2.MarketOrderArgsV2.__init__, token_id="t", amount=1.0, side=v2.Side.BUY, price=0.5, order_type="FAK")
    _bind(v2.PartialCreateOrderOptions.__init__, tick_size="0.01", neg_risk=False)
    _bind(v2.ApiCreds.__init__, api_key="a", api_secret="s", api_passphrase="p")
    _bind(v2.OrderPayload.__init__, orderID="x")
    _bind(v2.ClobClient.__init__, host="h", chain_id=137, key="k", signature_type=1, funder=None)

    args = v2.MarketOrderArgsV2(token_id="t", amount=1.0, side=v2.Side.BUY, price=0.5, order_type="FAK")
    _bind(
        v2.ClobClient.create_and_post_market_order,
        order_args=args,
        options=None,
        order_type=v2.OrderType.FAK,
    )
    _bind(v2.ClobClient.cancel_order, payload=v2.OrderPayload(orderID="x"))
    _bind(v2.ClobClient.get_balance_allowance)


def test_polymarket_sdk_adapter_matches_the_installed_sdk():
    """Every kwarg our SecureClient adapter sends must exist in the SDK."""
    sdk = pytest.importorskip("polymarket", reason=_SKIP)

    _bind(sdk.SecureClient.create, private_key="k", wallet=None)
    _bind(sdk.SecureClient.place_market_order, token_id="t", side="BUY", amount=1.0, max_price=0.5, order_type="FAK")
    _bind(sdk.SecureClient.place_market_order, token_id="t", side="SELL", shares=1.0, min_price=0.5, order_type="FAK")
    _bind(sdk.SecureClient.cancel_order, order_id="x")
    # The adapter passes asset_type explicitly because it is required.
    _bind(sdk.SecureClient.get_balance_allowance, asset_type="COLLATERAL")


def test_balance_extraction_matches_the_sdk_response_model():
    """The balance field our extractor reads must exist on the SDK model."""
    sdk = pytest.importorskip("polymarket", reason=_SKIP)
    fields = set(sdk.BalanceAllowance.model_fields)
    assert "balance" in fields
    # Amounts arrive in USDC base units, so the adapter scales by 1e6.
    assert LiveBroker._extract_balance({"balance": "2500000"}) == Decimal("2.5")
