"""Typed configuration read from environment variables.

Every value is read when `Config()` is instantiated, after `.env` has been
loaded, so `.env` always takes effect. See `polymarket_bot.env`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from .env import ensure_loaded, get_bool, get_float, get_int, get_str

GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"

_SECRET_FIELDS = {"private_key", "clob_api_key", "clob_api_secret", "clob_api_passphrase"}


class ConfigError(ValueError):
    """Raised when configuration is internally inconsistent."""


@dataclass
class Config:
    """Full bot configuration. Instantiate after `.env` is loaded."""

    # ---- Mode & credentials -------------------------------------------------
    mode: str = "paper"
    private_key: str = ""
    wallet: str = ""
    signature_type: int = 1
    clob_api_key: str = ""
    clob_api_secret: str = ""
    clob_api_passphrase: str = ""

    # ---- Endpoints ----------------------------------------------------------
    gamma_host: str = GAMMA_HOST
    clob_host: str = CLOB_HOST
    request_timeout: float = 10.0
    max_retries: int = 3

    # ---- Scan loop ----------------------------------------------------------
    scan_limit: int = 100
    poll_interval_seconds: int = 20
    max_markets_per_cycle: int = 60

    # ---- Complete-set arbitrage --------------------------------------------
    arb_enabled: bool = True
    arb_min_edge: float = 0.02
    arb_max_edge: float = 0.15
    arb_max_spread: float = 0.05
    arb_min_volume_24h: float = 10000.0
    arb_min_liquidity: float = 5000.0
    arb_min_seconds_left: int = 600
    arb_max_seconds_left: int = 604800
    arb_min_top_size: float = 100.0
    arb_cooldown_seconds: int = 60
    arb_antichase_spike: float = 0.08

    # ---- Basket arbitrage (neg-risk outcome sets) ---------------------------
    basket_enabled: bool = True
    basket_min_edge: float = 0.015
    basket_max_edge: float = 0.25
    basket_max_dislocation: float = 0.15
    basket_min_outcomes: int = 3
    basket_max_outcomes: int = 12
    basket_min_volume_24h: float = 25000.0
    basket_min_liquidity: float = 10000.0
    basket_min_top_size: float = 25.0
    basket_min_seconds_left: int = 3600

    # ---- Fade extreme -------------------------------------------------------
    fade_enabled: bool = False
    fade_price_below: float = 0.05
    fade_min_volume_24h: float = 50000.0
    fade_min_seconds_left: int = 3600
    fade_max_entry: float = 0.06
    fade_min_top_size: float = 250.0
    # Fraction of the way back toward 0.50 that a faded price is assumed to
    # travel. This is the strategy's core (unverifiable) assumption.
    fade_reversion_alpha: float = 0.10
    fade_min_edge: float = 0.01

    # ---- Exits --------------------------------------------------------------
    # Sell once the net liquidation value clears entry cost by this fraction.
    take_profit_pct: float = 0.15
    # Sell once the net liquidation value falls this fraction below entry cost.
    stop_loss_pct: float = 0.30
    exits_enabled: bool = True
    # Cap exposure across positions sharing a correlation theme (for example
    # several "Iran by September" markets). 0 disables the check.
    max_theme_exposure_usd: float = 150.0

    # ---- Risk management ----------------------------------------------------
    max_order_usd: float = 100.0
    max_total_exposure_usd: float = 500.0
    max_open_positions: int = 8
    max_orders_per_minute: int = 10
    kill_switch: bool = False
    min_free_balance_usd: float = 20.0
    # Reject a size whose average fill price is this far above the best ask,
    # i.e. do not walk deep enough through the book to move the market.
    max_book_impact: float = 0.03

    # ---- Fees ---------------------------------------------------------------
    # 0 means "use the per-market fee rate reported by the CLOB".
    taker_fee_rate_override: float = 0.0
    # Arbitrage legs are assumed to rest as maker orders when True, which
    # yields a zero fee and (optionally) a rebate. Default is taker.
    assume_maker: bool = False
    fee_safety_multiplier: float = 1.0

    # ---- Paper portfolio ----------------------------------------------------
    paper_starting_balance: float = 1000.0
    state_file: str = ""

    # ---- Execution ----------------------------------------------------------
    live_max_slippage: float = 0.005
    live_order_type: str = "FAK"

    # ---- Logging ------------------------------------------------------------
    log_level: str = "INFO"
    log_file: str = ""
    log_json: bool = False

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_env(cls, **overrides) -> "Config":
        """Build a Config from the environment, applying `overrides` last."""
        ensure_loaded()
        cfg = cls(
            mode=get_str("POLYMARKET_BOT_MODE", "paper").lower(),
            private_key=get_str("POLYMARKET_PRIVATE_KEY"),
            wallet=get_str("POLYMARKET_DEPOSIT_WALLET") or get_str("POLYMARKET_FUNDER_ADDRESS"),
            signature_type=get_int("POLYMARKET_SIGNATURE_TYPE", 1),
            clob_api_key=get_str("POLYMARKET_CLOB_API_KEY"),
            clob_api_secret=get_str("POLYMARKET_CLOB_API_SECRET"),
            clob_api_passphrase=get_str("POLYMARKET_CLOB_API_PASSPHRASE"),
            gamma_host=get_str("POLYMARKET_BOT_GAMMA_HOST", GAMMA_HOST),
            clob_host=get_str("POLYMARKET_BOT_CLOB_HOST", CLOB_HOST),
            request_timeout=get_float("POLYMARKET_BOT_REQUEST_TIMEOUT", 10.0),
            max_retries=get_int("POLYMARKET_BOT_MAX_RETRIES", 3),
            scan_limit=get_int("POLYMARKET_BOT_SCAN_LIMIT", 100),
            poll_interval_seconds=get_int("POLYMARKET_BOT_POLL_INTERVAL_S", 20),
            max_markets_per_cycle=get_int("POLYMARKET_BOT_MAX_MARKETS_PER_CYCLE", 60),
            arb_enabled=get_bool("POLYMARKET_BOT_ARB_ENABLED", True),
            arb_min_edge=get_float("POLYMARKET_BOT_ARB_MIN_EDGE", 0.02),
            arb_max_edge=get_float("POLYMARKET_BOT_ARB_MAX_EDGE", 0.15),
            arb_max_spread=get_float("POLYMARKET_BOT_ARB_MAX_SPREAD", 0.05),
            arb_min_volume_24h=get_float("POLYMARKET_BOT_ARB_MIN_VOLUME24H", 10000.0),
            arb_min_liquidity=get_float("POLYMARKET_BOT_ARB_MIN_LIQUIDITY", 5000.0),
            arb_min_seconds_left=get_int("POLYMARKET_BOT_ARB_MIN_SECONDS_LEFT", 600),
            arb_max_seconds_left=get_int("POLYMARKET_BOT_ARB_MAX_SECONDS_LEFT", 604800),
            arb_min_top_size=get_float("POLYMARKET_BOT_ARB_MIN_TOP_SIZE", 100.0),
            arb_cooldown_seconds=get_int("POLYMARKET_BOT_ARB_COOLDOWN_S", 60),
            arb_antichase_spike=get_float("POLYMARKET_BOT_ARB_ANTICHASE_SPIKE", 0.08),
            basket_enabled=get_bool("POLYMARKET_BOT_BASKET_ENABLED", True),
            basket_min_edge=get_float("POLYMARKET_BOT_BASKET_MIN_EDGE", 0.015),
            basket_max_edge=get_float("POLYMARKET_BOT_BASKET_MAX_EDGE", 0.25),
            basket_max_dislocation=get_float("POLYMARKET_BOT_BASKET_MAX_DISLOCATION", 0.15),
            basket_min_outcomes=get_int("POLYMARKET_BOT_BASKET_MIN_OUTCOMES", 3),
            basket_max_outcomes=get_int("POLYMARKET_BOT_BASKET_MAX_OUTCOMES", 12),
            basket_min_volume_24h=get_float("POLYMARKET_BOT_BASKET_MIN_VOLUME24H", 25000.0),
            basket_min_liquidity=get_float("POLYMARKET_BOT_BASKET_MIN_LIQUIDITY", 10000.0),
            basket_min_top_size=get_float("POLYMARKET_BOT_BASKET_MIN_TOP_SIZE", 25.0),
            basket_min_seconds_left=get_int("POLYMARKET_BOT_BASKET_MIN_SECONDS_LEFT", 3600),
            fade_enabled=get_bool("POLYMARKET_BOT_FADE_ENABLED", False),
            fade_price_below=get_float("POLYMARKET_BOT_FADE_BELOW", 0.05),
            fade_min_volume_24h=get_float("POLYMARKET_BOT_FADE_MIN_VOLUME24H", 50000.0),
            fade_min_seconds_left=get_int("POLYMARKET_BOT_FADE_MIN_SECONDS_LEFT", 3600),
            fade_max_entry=get_float("POLYMARKET_BOT_FADE_MAX_ENTRY", 0.06),
            fade_min_top_size=get_float("POLYMARKET_BOT_FADE_MIN_TOP_SIZE", 250.0),
            fade_reversion_alpha=get_float("POLYMARKET_BOT_FADE_REVERSION_ALPHA", 0.10),
            fade_min_edge=get_float("POLYMARKET_BOT_FADE_MIN_EDGE", 0.01),
            take_profit_pct=get_float("POLYMARKET_BOT_TAKE_PROFIT_PCT", 0.15),
            stop_loss_pct=get_float("POLYMARKET_BOT_STOP_LOSS_PCT", 0.30),
            exits_enabled=get_bool("POLYMARKET_BOT_EXITS_ENABLED", True),
            max_theme_exposure_usd=get_float("POLYMARKET_BOT_MAX_THEME_EXPOSURE_USD", 150.0),
            max_order_usd=get_float("POLYMARKET_BOT_MAX_ORDER_USD", 100.0),
            max_total_exposure_usd=get_float("POLYMARKET_BOT_MAX_TOTAL_EXPOSURE_USD", 500.0),
            max_open_positions=get_int("POLYMARKET_BOT_MAX_OPEN_POSITIONS", 8),
            max_orders_per_minute=get_int("POLYMARKET_BOT_MAX_ORDERS_PER_MIN", 10),
            kill_switch=get_bool("POLYMARKET_BOT_KILL_SWITCH", False),
            min_free_balance_usd=get_float("POLYMARKET_BOT_MIN_FREE_BALANCE_USD", 20.0),
            max_book_impact=get_float("POLYMARKET_BOT_MAX_BOOK_IMPACT", 0.03),
            taker_fee_rate_override=get_float("POLYMARKET_BOT_TAKER_FEE_RATE", 0.0),
            assume_maker=get_bool("POLYMARKET_BOT_ASSUME_MAKER", False),
            fee_safety_multiplier=get_float("POLYMARKET_BOT_FEE_SAFETY_MULTIPLIER", 1.0),
            paper_starting_balance=get_float("POLYMARKET_BOT_PAPER_BALANCE", 1000.0),
            state_file=get_str("POLYMARKET_BOT_STATE_FILE"),
            live_max_slippage=get_float("POLYMARKET_BOT_LIVE_MAX_SLIPPAGE", 0.005),
            live_order_type=get_str("POLYMARKET_BOT_LIVE_ORDER_TYPE", "FAK").upper(),
            log_level=get_str("POLYMARKET_BOT_LOG_LEVEL", "INFO").upper(),
            log_file=get_str("POLYMARKET_BOT_LOG_FILE"),
            log_json=get_bool("POLYMARKET_BOT_LOG_JSON", False),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        cfg.validate()
        return cfg

    # -------------------------------------------------------------- validation
    def validate(self) -> None:
        if self.mode not in {"paper", "live"}:
            raise ConfigError(f"POLYMARKET_BOT_MODE must be 'paper' or 'live', got {self.mode!r}")
        if self.live_order_type not in {"FAK", "FOK", "GTC", "GTD"}:
            raise ConfigError(f"unsupported live order type {self.live_order_type!r}")
        if self.signature_type not in {0, 1, 2, 3}:
            raise ConfigError("POLYMARKET_SIGNATURE_TYPE must be 0, 1, 2 or 3")
        for name, low, high in (
            ("arb_min_edge", 0.0, 1.0),
            ("arb_max_edge", 0.0, 1.0),
            ("arb_max_spread", 0.0, 1.0),
            ("basket_min_edge", 0.0, 1.0),
            ("basket_max_edge", 0.0, 1.0),
            ("fade_price_below", 0.0, 0.5),
            ("fade_max_entry", 0.0, 1.0),
            ("fade_reversion_alpha", 0.0, 1.0),
            ("fade_min_edge", 0.0, 1.0),
            ("max_book_impact", 0.0, 1.0),
        ):
            value = getattr(self, name)
            if not low <= value <= high:
                raise ConfigError(f"{name}={value} outside [{low}, {high}]")
        if self.arb_max_edge < self.arb_min_edge:
            raise ConfigError("arb_max_edge must be >= arb_min_edge")
        if self.basket_max_edge < self.basket_min_edge:
            raise ConfigError("basket_max_edge must be >= basket_min_edge")
        if self.arb_min_seconds_left > self.arb_max_seconds_left:
            raise ConfigError("arb_min_seconds_left must be <= arb_max_seconds_left")
        if self.basket_max_outcomes < self.basket_min_outcomes:
            raise ConfigError("basket_max_outcomes must be >= basket_min_outcomes")
        for name in ("max_order_usd", "max_total_exposure_usd", "paper_starting_balance"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be non-negative")
        if self.max_orders_per_minute < 1:
            raise ConfigError("max_orders_per_minute must be >= 1")
        if self.poll_interval_seconds < 1:
            raise ConfigError("poll_interval_seconds must be >= 1")
        if self.take_profit_pct < 0:
            raise ConfigError("take_profit_pct must be non-negative")
        if self.stop_loss_pct < 0:
            raise ConfigError("stop_loss_pct must be non-negative")
        if self.max_theme_exposure_usd < 0:
            raise ConfigError("max_theme_exposure_usd must be non-negative")

    # --------------------------------------------------------------- helpers
    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def has_credentials(self) -> bool:
        return bool(self.private_key)

    def describe(self) -> dict:
        """Human-readable summary with secrets redacted."""
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _SECRET_FIELDS:
                value = "<set>" if value else "<empty>"
            out[f.name] = value
        return out


def load_config(**overrides) -> Config:
    """Convenience wrapper around :meth:`Config.from_env`."""
    return Config.from_env(**overrides)
