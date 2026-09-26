"""Polymarket fee model.

Fees are charged on taker fills only, using the protocol formula::

    fee = shares x fee_rate x price x (1 - price)

The per-share fee therefore peaks at a price of 0.50 and falls away toward
both extremes. Maker orders pay nothing.

Two sources feed the fee rate:

1. The market's published category table (``docs.polymarket.com/trading/fees``),
   keyed by ``feeType``. This is the primary source because it distinguishes
   categories such as crypto (0.07) from politics (0.04).
2. The CLOB ``/fee-rate`` endpoint, which reports ``base_fee`` in basis
   points. Observed live values are only ``0`` or ``1000`` (10%), so this is
   used as a conservative fallback rather than a per-category signal.

``feesEnabled`` on the market object is the authoritative on/off switch: when
it is false, the market is fee-free regardless of its category.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .models import ZERO, to_decimal

# Category taker fee rates. Anything unmapped uses DEFAULT_TAKER_RATE.
CATEGORY_FEE_RATES: dict[str, Decimal] = {
    "crypto": Decimal("0.07"),
    "sports": Decimal("0.05"),
    "economics": Decimal("0.05"),
    "culture": Decimal("0.05"),
    "weather": Decimal("0.05"),
    "other": Decimal("0.05"),
    "finance": Decimal("0.04"),
    "politics": Decimal("0.04"),
    "mentions": Decimal("0.04"),
    "tech": Decimal("0.04"),
    "geopolitics": Decimal("0"),
}

# Prefixes seen on Gamma `feeType` values (for example "sports_fees_v3",
# "crypto_fees_v2", "finance_prices_fees", "zero_fees").
_FEE_TYPE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("zero", "geopolitics"),
    ("geopolitic", "geopolitics"),
    ("crypto", "crypto"),
    ("sports", "sports"),
    ("economics", "economics"),
    ("culture", "culture"),
    ("weather", "weather"),
    ("finance", "finance"),
    ("politics", "politics"),
    ("mentions", "mentions"),
    ("tech", "tech"),
)

DEFAULT_TAKER_RATE = Decimal("0.05")


def category_from_fee_type(fee_type: str | None) -> str | None:
    """Map a Gamma ``feeType`` such as ``sports_fees_v3`` to a fee category."""
    if not fee_type:
        return None
    normalized = str(fee_type).strip().lower()
    for prefix, category in _FEE_TYPE_PREFIXES:
        if normalized.startswith(prefix):
            return category
    return None


def resolve_taker_rate(
    *,
    fees_enabled: bool,
    fee_type: str | None = None,
    base_fee_bps: int | None = None,
    override: float | Decimal | None = None,
) -> Decimal:
    """Determine the taker fee rate for a market.

    Priority: explicit override, then fee-free market, then the category
    table, then the CLOB-reported base fee, then the default.
    """
    if override:
        value = to_decimal(override)
        if value > 0:
            return value
    if not fees_enabled:
        return ZERO
    category = category_from_fee_type(fee_type)
    if category is not None:
        return CATEGORY_FEE_RATES.get(category, DEFAULT_TAKER_RATE)
    if base_fee_bps is not None:
        # base_fee is reported in basis points and observed only as 0 or 1000.
        # Treating 1000 bps as 10% would badly overstate real fees, so clamp
        # to the highest known category rate instead.
        if base_fee_bps <= 0:
            return ZERO
        return DEFAULT_TAKER_RATE
    return DEFAULT_TAKER_RATE


def taker_fee(shares: Decimal, price: Decimal, rate: Decimal) -> Decimal:
    """Protocol taker fee in USDC for one fill."""
    if shares <= 0 or price <= 0 or rate <= 0:
        return ZERO
    return shares * rate * price * (Decimal(1) - price)


@dataclass(frozen=True)
class FeeModel:
    """Fee parameters for a single market, ready for cost calculations."""

    taker_rate: Decimal = ZERO
    maker_rate: Decimal = ZERO
    rebate_share: Decimal = ZERO
    safety_multiplier: Decimal = Decimal(1)

    @property
    def is_fee_free(self) -> bool:
        return self.taker_rate <= 0

    @classmethod
    def for_market(
        cls,
        *,
        fees_enabled: bool,
        fee_type: str | None = None,
        base_fee_bps: int | None = None,
        override: float | Decimal | None = None,
        safety_multiplier: float | Decimal = 1,
        maker: bool = False,
    ) -> "FeeModel":
        rate = resolve_taker_rate(
            fees_enabled=fees_enabled,
            fee_type=fee_type,
            base_fee_bps=base_fee_bps,
            override=override,
        )
        multiplier = to_decimal(safety_multiplier, Decimal(1))
        if multiplier <= 0:
            multiplier = Decimal(1)
        if maker:
            rate = ZERO
        return cls(
            taker_rate=rate,
            maker_rate=ZERO,
            rebate_share=ZERO,
            safety_multiplier=multiplier,
        )

    def buy_fee(self, shares: Decimal, price: Decimal) -> Decimal:
        """Fee to buy `shares` at `price`."""
        return taker_fee(shares, price, self.taker_rate) * self.safety_multiplier

    def sell_fee(self, shares: Decimal, price: Decimal) -> Decimal:
        return taker_fee(shares, price, self.taker_rate) * self.safety_multiplier

    def fee_for_notional(self, usd: Decimal, price: Decimal) -> Decimal:
        """Fee for spending `usd` at `price` (derives the share count)."""
        if price <= 0:
            return ZERO
        return self.buy_fee(usd / price, price)

    def marginal_rate_at(self, price: Decimal) -> Decimal:
        """Fee per additional share at `price` (derivative of the fee curve)."""
        if price <= 0 or self.taker_rate <= 0:
            return ZERO
        return self.taker_rate * (Decimal(1) - price) * self.safety_multiplier
