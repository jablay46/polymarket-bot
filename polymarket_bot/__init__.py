"""Polymarket trading bot — multi-strategy, risk-managed, paper-first.

Strategies included:
  * complete-set arbitrage within a single binary market
  * basket arbitrage across a mutually exclusive outcome set (neg-risk)
  * fade-extreme mean reversion on very liquid markets

The default mode is paper trading. Live trading requires an explicit
POLYMARKET_BOT_MODE=live plus wallet credentials.
"""

__version__ = "1.0.0"
