"""Order placement backends.

``PaperBroker`` fills instantly at the modelled price and moves no real money.

``LiveBroker`` talks to Polymarket through an official client. Two backends
are supported and probed in order:

1. ``polymarket-client`` (import name ``polymarket``) — the current official
   SDK, with ``SecureClient.create(private_key=..., wallet=...)`` deriving API
   credentials automatically.
2. ``py-clob-client-v2`` — the previous official client, still widely used.

Both are imported lazily so paper mode needs no third-party dependency. The
adapter surface is deliberately small: balance, buy, sell, cancel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .config import Config
from .logging_setup import get_logger
from .models import ONE, ZERO, to_decimal

log = get_logger("broker")


class BrokerError(RuntimeError):
    """Raised when an order cannot be submitted."""


@dataclass
class LegResult:
    """Outcome of a single order submission."""

    token_id: str
    side: str
    ok: bool
    order_id: str = ""
    status: str = ""
    filled_shares: Decimal = ZERO
    filled_usd: Decimal = ZERO
    error: str = ""
    raw: dict = field(default_factory=dict)


class PaperBroker:
    """Deterministic fill simulator.

    This broker holds no cash of its own. The :class:`~polymarket_bot.portfolio.Portfolio`
    is the single source of truth for paper balances, so cash is never debited
    twice.
    """

    name = "paper"
    supports_balance = False

    def __init__(self, config: Config):
        self.config = config
        self.orders: list[dict] = []

    def balance_usd(self) -> Decimal:
        return ZERO

    def buy(
        self,
        *,
        token_id: str,
        usd: Decimal,
        price: Decimal,
        tick_size: Decimal = Decimal("0.01"),
        neg_risk: bool = False,
        order_type: str = "FAK",
    ) -> LegResult:
        if usd <= 0 or price <= 0:
            return LegResult(token_id, "BUY", False, error="invalid amount or price")
        shares = usd / price
        self.orders.append({"token_id": token_id, "side": "BUY", "shares": str(shares), "usd": str(usd), "price": str(price)})
        return LegResult(
            token_id=token_id,
            side="BUY",
            ok=True,
            order_id=f"paper-{len(self.orders)}",
            status="matched",
            filled_shares=shares,
            filled_usd=usd,
        )

    def sell(
        self,
        *,
        token_id: str,
        shares: Decimal,
        price: Decimal,
        tick_size: Decimal = Decimal("0.01"),
        neg_risk: bool = False,
        order_type: str = "FAK",
    ) -> LegResult:
        if shares <= 0 or price <= 0:
            return LegResult(token_id, "SELL", False, error="invalid amount or price")
        usd = shares * price
        self.orders.append({"token_id": token_id, "side": "SELL", "shares": str(shares), "usd": str(usd), "price": str(price)})
        return LegResult(
            token_id=token_id,
            side="SELL",
            ok=True,
            order_id=f"paper-{len(self.orders)}",
            status="matched",
            filled_shares=shares,
            filled_usd=usd,
        )

    def cancel(self, order_id: str) -> bool:
        return True


class LiveBroker:
    """Order placement through an official Polymarket client."""

    name = "live"
    supports_balance = True

    def __init__(self, config: Config):
        self.config = config
        self._client = None
        self._backend = ""
        self._connect()

    # ------------------------------------------------------------- connect
    def _connect(self) -> None:
        if not self.config.private_key:
            raise BrokerError("live mode requires POLYMARKET_PRIVATE_KEY")
        errors: list[str] = []
        for backend, factory in (("polymarket-client", self._connect_sdk), ("py-clob-client-v2", self._connect_v2)):
            try:
                client = factory()
            except ImportError as exc:
                errors.append(f"{backend}: not installed ({exc})")
                continue
            except Exception as exc:  # noqa: BLE001 - surface any auth failure
                errors.append(f"{backend}: {exc}")
                continue
            self._client = client
            self._backend = backend
            log.info("live broker connected via %s", backend)
            return
        raise BrokerError("no usable CLOB client. Install one of: " + "; ".join(errors))

    def _connect_sdk(self):
        from polymarket import SecureClient  # noqa: PLC0415 - optional dependency

        return SecureClient.create(
            private_key=self.config.private_key,
            wallet=self.config.wallet or None,
        )

    def _connect_v2(self):
        from py_clob_client_v2 import ClobClient  # noqa: PLC0415 - optional dependency

        client = ClobClient(
            host=self.config.clob_host,
            chain_id=137,
            key=self.config.private_key,
            signature_type=self.config.signature_type,
            funder=self.config.wallet or None,
        )
        creds = None
        if self.config.clob_api_key:
            from py_clob_client_v2 import ApiCreds  # noqa: PLC0415

            creds = ApiCreds(
                api_key=self.config.clob_api_key,
                api_secret=self.config.clob_api_secret,
                api_passphrase=self.config.clob_api_passphrase,
            )
        else:
            creds = client.create_or_derive_api_key()
        client.set_api_creds(creds)
        return client

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------- balance
    def balance_usd(self) -> Decimal:
        """Free collateral balance, or zero if the venue cannot report it."""
        try:
            if self._backend == "polymarket-client":
                # SecureClient requires the asset type to be named explicitly.
                value = self._client.get_balance_allowance(asset_type="COLLATERAL")
                return self._extract_balance(value)
            value = self._client.get_balance_allowance()
            return self._extract_balance(value)
        except Exception as exc:  # noqa: BLE001 - balance is advisory
            log.warning("could not read balance: %s", exc)
            return ZERO

    @staticmethod
    def _extract_balance(value) -> Decimal:
        if value is None:
            return ZERO
        for attr in ("balance", "available", "collateral", "usdc_balance"):
            if hasattr(value, attr):
                return to_decimal(getattr(value, attr)) / Decimal(10**6)
        if isinstance(value, dict):
            for key in ("balance", "available", "collateral"):
                if key in value:
                    return to_decimal(value[key]) / Decimal(10**6)
        return ZERO

    # ----------------------------------------------------------------- buy
    def buy(
        self,
        *,
        token_id: str,
        usd: Decimal,
        price: Decimal,
        tick_size: Decimal = Decimal("0.01"),
        neg_risk: bool = False,
        order_type: str = "FAK",
    ) -> LegResult:
        if usd <= 0:
            return LegResult(token_id, "BUY", False, error="non-positive notional")
        max_price = min(ONE, price * (ONE + Decimal(str(self.config.live_max_slippage))))
        try:
            if self._backend == "polymarket-client":
                resp = self._client.place_market_order(
                    token_id=token_id,
                    side="BUY",
                    amount=usd,
                    max_price=max_price,
                    order_type=order_type if order_type in {"FAK", "FOK"} else "FAK",
                )
            else:
                resp = self._buy_v2(token_id, usd, max_price, tick_size, neg_risk, order_type)
        except Exception as exc:  # noqa: BLE001 - network and venue errors
            return LegResult(token_id, "BUY", False, error=str(exc))
        return self._interpret(resp, token_id, "BUY")

    def _buy_v2(self, token_id, usd, max_price, tick_size, neg_risk, order_type):
        from py_clob_client_v2 import (  # noqa: PLC0415
            MarketOrderArgsV2,
            OrderType,
            PartialCreateOrderOptions,
            Side,
        )

        options = PartialCreateOrderOptions(tick_size=str(tick_size), neg_risk=neg_risk)
        args = MarketOrderArgsV2(
            token_id=token_id, amount=float(usd), side=Side.BUY, price=float(max_price), order_type=order_type
        )
        return self._client.create_and_post_market_order(args, options=options, order_type=OrderType.FAK)

    # ---------------------------------------------------------------- sell
    def sell(
        self,
        *,
        token_id: str,
        shares: Decimal,
        price: Decimal,
        tick_size: Decimal = Decimal("0.01"),
        neg_risk: bool = False,
        order_type: str = "FAK",
    ) -> LegResult:
        if shares <= 0:
            return LegResult(token_id, "SELL", False, error="non-positive size")
        min_price = max(ZERO, price * (ONE - Decimal(str(self.config.live_max_slippage))))
        try:
            if self._backend == "polymarket-client":
                resp = self._client.place_market_order(
                    token_id=token_id,
                    side="SELL",
                    shares=shares,
                    min_price=min_price,
                    order_type=order_type if order_type in {"FAK", "FOK"} else "FAK",
                )
            else:
                resp = self._sell_v2(token_id, shares, min_price, tick_size, neg_risk)
        except Exception as exc:  # noqa: BLE001
            return LegResult(token_id, "SELL", False, error=str(exc))
        return self._interpret(resp, token_id, "SELL")

    def _sell_v2(self, token_id, shares, min_price, tick_size, neg_risk):
        from py_clob_client_v2 import (  # noqa: PLC0415
            MarketOrderArgsV2,
            OrderType,
            PartialCreateOrderOptions,
            Side,
        )

        options = PartialCreateOrderOptions(tick_size=str(tick_size), neg_risk=neg_risk)
        args = MarketOrderArgsV2(
            token_id=token_id, amount=float(shares), side=Side.SELL, price=float(min_price), order_type="FAK"
        )
        return self._client.create_and_post_market_order(args, options=options, order_type=OrderType.FAK)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _interpret(resp, token_id: str, side: str) -> LegResult:
        if resp is None:
            return LegResult(token_id, side, False, error="empty response")
        ok = bool(getattr(resp, "ok", None) if hasattr(resp, "ok") else (resp.get("success") if isinstance(resp, dict) else False))
        if not ok:
            message = getattr(resp, "message", None) or getattr(resp, "errorMsg", None)
            if message is None and isinstance(resp, dict):
                message = resp.get("errorMsg") or resp.get("message")
            code = getattr(resp, "code", "") if hasattr(resp, "code") else ""
            return LegResult(token_id, side, False, error=f"{code} {message}".strip(), raw=_as_dict(resp))
        order_id = getattr(resp, "order_id", None) or (resp.get("orderID") if isinstance(resp, dict) else "") or ""
        status = getattr(resp, "status", None) or (resp.get("status") if isinstance(resp, dict) else "") or ""
        making = to_decimal(getattr(resp, "making_amount", None) or (resp.get("makingAmount") if isinstance(resp, dict) else 0))
        taking = to_decimal(getattr(resp, "taking_amount", None) or (resp.get("takingAmount") if isinstance(resp, dict) else 0))
        if side == "BUY":
            usd, shares = making, taking
        else:
            shares, usd = making, taking
        return LegResult(
            token_id=token_id,
            side=side,
            ok=True,
            order_id=str(order_id),
            status=str(status),
            filled_shares=shares,
            filled_usd=usd,
            raw=_as_dict(resp),
        )

    def cancel(self, order_id: str) -> bool:
        try:
            if self._backend == "polymarket-client":
                self._client.cancel_order(order_id=order_id)
            else:
                from py_clob_client_v2 import OrderPayload  # noqa: PLC0415

                self._client.cancel_order(OrderPayload(orderID=order_id))
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("cancel failed for %s: %s", order_id, exc)
            return False

    def close(self) -> None:
        closer = getattr(self._client, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass


def _as_dict(resp) -> dict:
    if isinstance(resp, dict):
        return resp
    model_dump = getattr(resp, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:  # noqa: BLE001
            return {}
    return {}


def build_broker(config: Config):
    """Construct the broker implied by the configuration."""
    if config.is_live:
        return LiveBroker(config)
    return PaperBroker(config)
