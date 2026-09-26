# polymarket-bot

An arbitrage-first Polymarket bot in pure Python. It scans live markets,
finds mispriced complete sets, sizes each trade against real order-book
depth, and executes through a broker you can swap between paper and live.

- **No dependencies for paper mode.** Standard library only.
- **Arbitrage first.** Two risk-free strategies enabled by default.
- **Honest numbers.** Fees, slippage, and book depth are modelled per leg.
- **113 tests**, no network required to run them.

```
$ python run.py doctor
doctor: PASS

$ python run.py scan
scanned 111 market group(s)
no opportunities above the configured thresholds.
```

That last output is the point. Polymarket's complete sets are usually
priced at or above $1.00, so a correct scanner reports nothing most of the
time. A bot that always finds "opportunities" is measuring its own bug.

---

## Quick start

```bash
git clone https://github.com/jablay46/polymarket-bot
cd polymarket-bot

python run.py doctor        # check config, network, and fees
python run.py scan          # one read-only pass, places nothing
python run.py run           # continuous paper trading loop
```

Python 3.11 or newer. No `pip install` is needed to paper trade.

## What it trades

### 1. Complete-set arbitrage (risk-free)

On a binary market, exactly one of Yes and No pays $1 at resolution. If you
can buy both for less than $1, you have locked in the difference:

```
buy  Yes @ 0.47
buy  No  @ 0.50
cost      0.97  ->  payout 1.00  =  +0.03 per set
```

The bot walks both order books level by level rather than trusting the top
quote, so the size it reports is the size the book can actually fill. It
then binary-searches for the largest size that stays profitable.

### 2. Basket arbitrage (risk-free)

The same idea for multi-outcome events such as "EPL 2027 Champion". Exactly
one of the N outcomes resolves Yes, so buying every Yes for a combined
price below $1 is again risk-free. Sizing uses a monotone search over the
joint cost curve.

### 3. Fade extreme (directional, off by default)

Buys very cheap longshots on liquid markets, betting the price drifts back
toward 0.50. **This is not arbitrage.** The edge is an assumption baked into
`POLYMARKET_BOT_FADE_REVERSION_ALPHA`, not a market dislocation, and the bot
labels every such position as directional. Its assumed profit is reported
separately from locked-in arbitrage profit so the two can never be confused.

Leave it off unless you have your own reason to believe the reversion.

## How the money is tracked

The paper ledger separates two very different numbers:

```
hedged_profit=$12.40        certain, from complete sets
directional=$50.00 (assumed_profit=$310.00)
```

`hedged_profit` is money you have locked in. `assumed_profit` is a
model output that may never happen. Only the first is real.

## Commands

| Command | What it does |
| --- | --- |
| `run.py doctor` | Validate config, reach the API, probe fees and credentials. |
| `run.py scan` | One read-only pass. Prints opportunities, places nothing. |
| `run.py run` | Continuous loop. Paper by default. |
| `run.py config` | Print the effective configuration. |

Useful flags:

```bash
python run.py run --max-cycles 5          # stop after 5 passes
python run.py run --kill-switch           # decide, but never place an order
python run.py run --mode live             # requires credentials
python run.py scan --limit 300 --json     # more markets, machine-readable
python run.py scan --show-all             # include groups with no signal
```

Strategies are toggled with `POLYMARKET_BOT_<NAME>_ENABLED`, not a flag.

## Configuration

Everything is an environment variable, documented with defaults in
[`.env.example`](.env.example):

```bash
cp .env.example .env
python run.py scan
```

The ones that matter most:

| Variable | Default | Meaning |
| --- | --- | --- |
| `POLYMARKET_BOT_MODE` | `paper` | `paper` or `live` |
| `POLYMARKET_BOT_ARB_MIN_EDGE` | `0.02` | Minimum net edge per set |
| `POLYMARKET_BOT_MAX_ORDER_USD` | `100` | Per-order notional cap |
| `POLYMARKET_BOT_MAX_TOTAL_EXPOSURE_USD` | `500` | Total capital at risk |
| `POLYMARKET_BOT_MAX_OPEN_POSITIONS` | `8` | Concurrent positions |
| `POLYMARKET_BOT_MIN_FREE_BALANCE_USD` | `20` | Cash kept in reserve |
| `POLYMARKET_BOT_KILL_SWITCH` | `false` | Detect but never trade |

## Going live

Live trading is opt-in and refuses to start without credentials.

```bash
pip install py-clob-client-v2      # or: pip install polymarket-client

export POLYMARKET_BOT_MODE=live
export POLYMARKET_PRIVATE_KEY=0x...
export POLYMARKET_DEPOSIT_WALLET=0x...
python run.py doctor               # confirm credentials before trading
python run.py run
```

The broker auto-detects either official client, so you do not have to pick
one. Before risking real money, at minimum:

1. Run `scan` for a day and confirm it only fires when the maths is right.
2. Set `POLYMARKET_BOT_MAX_ORDER_USD` and `MAX_TOTAL_EXPOSURE_USD` low.
3. Start with `POLYMARKET_BOT_KILL_SWITCH=true` to watch decisions without
   placing orders.

## Safety properties

These are enforced in code, not by convention:

- **No double spending.** The portfolio is the single source of truth for
  paper cash; the paper broker never debits anything itself.
- **Depth-aware sizing.** A trade is capped by what the book can fill, by
  `max_book_impact`, by per-order and total exposure, and by free cash.
- **Leg-risk unwind.** If one leg of a two-leg arbitrage fails, the bot
  immediately unwinds the filled leg instead of holding a naked position.
- **Anti-chase.** A market whose price spiked since the last scan is
  skipped, so the bot does not buy into a moving quote.
- **Fee realism.** Fees come from the venue per market, with an optional
  safety multiplier, and are subtracted before any edge is reported.

## Layout

```
polymarket_bot/
  models.py       order books, market groups, signals, depth walks
  data.py         Gamma + CLOB clients, market discovery
  fees.py         per-market fee model
  strategies.py   the three strategies
  risk.py         sizing, exposure, rate limits, kill switch
  execution.py    leg ordering and failure unwind
  brokers.py      paper and live order placement
  portfolio.py    cash ledger and positions
  engine.py       the scan/decide/execute loop
  cli.py          command line interface
tests/            113 tests, no network
```

## Tests

```bash
python -m pytest -q
```

The suite runs offline and covers book walking, fee maths, each strategy's
edge cases, risk sizing, execution unwind, and the engine loop.

## Disclaimer

This is trading software, provided as-is under the MIT licence. Nothing here
is financial advice. Arbitrage windows are rare and short-lived, fees and
slippage can erase small edges, and you can lose money. Test in paper mode
first and size your risk deliberately.
