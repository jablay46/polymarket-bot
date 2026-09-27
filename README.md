# polymarket-bot

An arbitrage-first Polymarket bot in pure Python. It scans live markets,
finds mispriced complete sets, sizes each trade against real order-book
depth, and executes through a broker you can swap between paper and live.

- **No dependencies for paper mode.** Standard library only.
- **Arbitrage first.** Two risk-free strategies enabled by default.
- **Honest numbers.** Fees, slippage, and book depth are modelled per leg.
- **202 tests**, no network required to run them.

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

### 4. Cross-market threshold ladders (off by default)

Some questions form a numeric ladder on one subject and window — "Will BTC be
above $90k by Dec 31?" and "Will BTC be above $100k by Dec 31?". Above the
higher threshold implies above the lower, so the two are not independent, and
when the market prices them out of order the safe side pays $1 in every state
the relation allows.

This is the one strategy here whose edge rests on a *relation the code guessed
at*, not on something the CLOB guarantees. A binary market's Yes+No partition
$1 by construction; a neg-risk event's outcomes are mutually exclusive by
protocol. A ladder relation is inferred from the question text, and the text
cannot distinguish "BTC above $90k, Binance spot" from "BTC above $90k,
Coinbase spot" — questions that look identical to the parser but are not the
same claim. When the relation does not actually hold, the "arbitrage" is a
directional bet wearing an arbitrage label.

The strategy is off by default and gated accordingly:

* **Paper mode** prices any detected ladder, and each signal carries
  `metadata["unverified_relation"] = True` and `metadata["confirmed_pair"]`.
  Paper fills are how you find out whether a relation is worth confirming.
* **Live mode** trades a ladder only when its two condition ids appear in
  `POLYMARKET_BOT_CROSS_MARKET_CONFIRMED_PAIRS`. That list is empty by
  default, so live mode trades no ladders at all until a human puts a pair
  there. Run `python run.py ladders` to list candidates, then open both
  markets and confirm the resolution rules agree before listing a pair.

Two details that are easy to get wrong, and are load-bearing:

* **Which two legs.** On an "above" ladder the implication is
  `above(high) => above(low)`, so the safe side is **Yes(low) + No(high)**. A
  "below" ladder implies the other way and needs the mirror pair,
  **Yes(high) + No(low)**. The two are not interchangeable; picking the wrong
  one turns a covered position into a bet against the middle of the range.
* **Direction is part of the group key.** "Above $90k" and "below $100k" on
  one subject are not ordered against each other, so grouping them would price
  noise as edge.

## How the money is tracked

The paper ledger separates two very different numbers:

```
hedged_profit=$12.40        certain, from complete sets
directional=$50.00 (assumed_profit=$310.00)
```

`hedged_profit` is money you have locked in. `assumed_profit` is a
model output that may never happen. Only the first is real.

## Closing positions

Positions do not sit until the market dies. Every cycle, before it looks for
new trades, the bot works its open positions:

| Path | Trigger | Result |
| --- | --- | --- |
| Take profit | Net sale value clears entry cost by `TAKE_PROFIT_PCT` | Sells into the bids |
| Stop loss | Net sale value falls `STOP_LOSS_PCT` below entry cost | Cuts the loss |
| Settlement | Market resolved | Pays $1 per winning share, no order needed |

Two properties keep this honest:

* **Exits are priced against the live book.** The quoted profit is what the
  resting bids would actually pay for the whole size, net of taker fees. If the
  book is too thin to absorb the position, the exit is skipped and retried
  later, because a paper profit nobody will buy is not a profit.
* **A hedged set exits all-or-nothing.** Selling one leg of a pair would leave
  naked directional risk, so every leg must be fully sellable or nothing is
  sold. Directional (fade) positions exit leg by leg, since there is no hedge
  to break.
* **A partial exit keeps the remainder on the books.** If a leg fails or fills
  short, the shares that sold are booked and the rest stay open, with their
  cost basis intact. Deleting the position there would hide risk the bot still
  holds and overstate the exposure headroom available for new trades.

A market can only be bought once. The per-market cooldown is just a timer and
will always expire, so a separate guard refuses any entry while a position on
that market is still open. Closing a position frees the market again.

The one exception is a **stop-loss**. Because the exit pass runs before the
entry pass, a stopped position would otherwise free its own slot and the same
signal would be re-opened on the very next line — one bad trade billed four
times. A stopped market is therefore blocked from re-entry for
`POLYMARKET_BOT_REENTRY_COOLDOWN_S` (15 minutes by default). That market's
thesis just failed; the timer is there so the bot stops mistaking "the slot is
free" for "the trade is still good".

## Commands

| Command | What it does |
| --- | --- |
| `run.py doctor` | Validate config, reach the API, probe fees and credentials. |
| `run.py scan` | One read-only pass. Prints opportunities, places nothing. |
| `run.py ladders` | List detected threshold ladders and their pair ids. Read-only. |
| `run.py run` | Continuous loop. Paper by default. |
| `run.py config` | Print the effective configuration. |

Useful flags:

```bash
python run.py run --max-cycles 5          # stop after 5 passes
python run.py run --kill-switch           # decide, but never place an order
python run.py run --mode live             # requires credentials
python run.py scan --limit 300 --json     # more markets, machine-readable
python run.py scan --show-all             # include groups with no signal
python run.py ladders --limit 300         # search deeper for ladder markets
```

Strategies are toggled with `POLYMARKET_BOT_<NAME>_ENABLED`, not a flag.

## Troubleshooting

**`doctor` fails with `CERTIFICATE_VERIFY_FAILED` / hostname mismatch on a host
that works everywhere else.** Something between you and Polymarket is
terminating TLS with a certificate the system trust store does not know — a
corporate proxy, a firewall, or a captive portal. This is a network problem, not
a bot problem, and `doctor` names it as such. Fix it at the network layer:

- Add the proxy's root CA to the system trust store, or
- point the bot at the proxy's root CA with
  `POLYMARKET_BOT_CA_BUNDLE=/path/to/proxy-ca.pem`.

Do not disable certificate verification to work around this. The bot has no
option to do so on purpose; without a valid certificate you cannot tell the
exchange from an attacker, and this is the network that carries your orders.

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
| `POLYMARKET_BOT_TAKE_PROFIT_PCT` | `0.15` | Profit at which a position is sold |
| `POLYMARKET_BOT_STOP_LOSS_PCT` | `0.30` | Loss at which a position is cut |
| `POLYMARKET_BOT_MAX_THEME_EXPOSURE_USD` | `150` | Cap shared by correlated markets |
| `POLYMARKET_BOT_REENTRY_COOLDOWN_S` | `900` | No re-buy of a stopped market for this long |
| `POLYMARKET_BOT_FADE_MAX_ROUND_TRIP_RATIO` | `0.25` | Max spread+fees for a fade, as a fraction of entry |
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
- **One position per market.** A held market is refused outright, so the
  bot can never average into a bet it already owns.
- **Correlation cap.** Markets sharing a story (several Iran questions) draw
  from one shared budget instead of stacking the same bet.
- **Bounded downside.** A stop loss cuts directional bets that go wrong, and
  every exit is validated against the live book before it is sent.
- **Fee realism.** Fees come from the venue per market, with an optional
  safety multiplier, and are subtracted before any edge is reported.
- **Guessed relations stay gated.** Cross-market ladders rest on a relation
  inferred from question text. Live mode refuses every pair not named in
  `POLYMARKET_BOT_CROSS_MARKET_CONFIRMED_PAIRS`, and that list starts empty.
- **Per-leg venue settings.** Each leg of a cross-market position carries its
  own condition id, tick size, and neg-risk flag, so entry, unwind, and
  settlement are all sent to the market the leg actually belongs to.

## Layout

```
polymarket_bot/
  models.py       order books, market groups, signals, depth walks
  data.py         Gamma + CLOB clients, market discovery
  fees.py         per-market fee model
  strategies.py   set-arb, basket, and fade strategies
  relations.py    threshold-ladder detection from question text
  cross_market.py cross-market ladder pricing and live gate
  risk.py         sizing, exposure, rate limits, kill switch
  execution.py    leg ordering and failure unwind
  exits.py        take profit, stop loss, settlement
  brokers.py      paper and live order placement
  portfolio.py    cash ledger and positions
  engine.py       the scan/decide/execute loop
  cli.py          command line interface
tests/            196 tests, no network
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
