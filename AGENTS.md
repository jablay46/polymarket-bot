# polymarket-bot — agent notes

Paper-first Polymarket bot. Standard library only in paper mode; `pytest` for tests.

## Commands

- Run paper mode: `python3 run.py run [--max-cycles N]`
- Full tests: `python3 -m pytest -q` (no network required)
- Inspect detected ladders: `python3 run.py ladders [--json]`

## Configuration

- `.env` in the working directory is loaded automatically at import
  (`polymarket_bot/env.py:ensure_loaded`). Precedence, highest first:
  `--env-file FILE` (override=True) > shell environment > `./.env`.
  A stale `./.env` from an old `.env.example` is the usual reason a strategy
  is on when the repo default says off (e.g. `fade=True`).
- Defaults: `arb=True`, `basket=True`, `fade=False`, `cross_market=False`.
- Target paper config: arb + basket ON, fade OFF. Fade is the only strategy that
  opens *directional* positions (unhedged model bets); it is the main source of
  fills in a quiet market, so expect 0 signals with it off.
- TLS failures are almost always an inactive venv, not a bad CA. `run.py doctor`
  reports the served certificate; `POLYMARKET_BOT_CA_BUNDLE` overrides the CA file.

## Key invariants

- Gamma does **not** guarantee that `clobTokenIds[0]` is Yes. Resolve Yes by
  outcome name (`MarketInfo.yes_index`); never assume index. Getting this wrong
  swaps the two legs of a cross-market pair and prices a directional bet as an
  arbitrage. See `polymarket_bot/cross_market.py:_resolve_yes_index`.
- Ladder relation: buy Yes(implied) + No(implying). "above" and "below" ladders
  use opposite legs; never mix directions in one group.
- Cross-market relations are keyword-guessed, not verified. Live mode refuses any
  pair not named in `POLYMARKET_BOT_CROSS_MARKET_CONFIRMED_PAIRS` (empty default).
- Threshold regex: k/m suffix must end on a word boundary or "the $90k *m*ark"
  parses as 90 billion. Double negatives ("not less than") mean *above*.
- Directional positions are unhedged model bets; `assumed_profit` is not locked.
  `equity` is cost-basis, so it does not move with mark-to-market until exit.
