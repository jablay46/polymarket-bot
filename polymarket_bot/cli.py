"""Command-line interface.

Commands::

    polymarket-bot run      # trade (paper by default)
    polymarket-bot scan     # one-shot opportunity scan, no orders
    polymarket-bot doctor   # validate configuration and connectivity
    polymarket-bot config   # print effective configuration (secrets redacted)
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal

from . import __version__
from .config import Config, ConfigError
from .data import MarketScanner
from .logging_setup import get_logger, setup_logging
from .models import ZERO

log = get_logger("cli")

STRATEGY_LABELS = {
    "set_arbitrage": "complete-set arbitrage",
    "basket_arbitrage": "basket arbitrage (neg-risk)",
    "fade_extreme": "fade extreme (directional)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polymarket-bot",
        description="Polymarket trading bot: set arbitrage, basket arbitrage, fade extreme, cross-market ladders.",
    )
    parser.add_argument("--version", action="version", version=f"polymarket-bot {__version__}")
    parser.add_argument("--env-file", default=None, help="path to a .env file (default: ./.env)")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run the trading loop")
    run.add_argument("--mode", choices=["paper", "live"], help="override POLYMARKET_BOT_MODE")
    run.add_argument("--max-cycles", type=int, default=None, help="stop after N cycles")
    run.add_argument("--kill-switch", action="store_true", help="detect signals but never execute")

    scan = sub.add_parser("scan", help="scan once and print opportunities")
    scan.add_argument("--limit", type=int, default=None, help="markets to scan")
    scan.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    scan.add_argument("--show-all", action="store_true", help="also print groups with no signal")

    sub.add_parser("doctor", help="validate configuration and connectivity")
    sub.add_parser("config", help="print the effective configuration")

    ladders = sub.add_parser(
        "ladders",
        help="list detected cross-market threshold ladders and their pair ids",
    )
    ladders.add_argument("--limit", type=int, default=None, help="markets to scan")
    ladders.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def _load(args) -> Config:
    from .env import load_dotenv

    if args.env_file:
        load_dotenv(args.env_file, override=True)
    overrides = {}
    if getattr(args, "mode", None):
        overrides["mode"] = args.mode
    if getattr(args, "kill_switch", False):
        overrides["kill_switch"] = True
    if getattr(args, "limit", None):
        overrides["scan_limit"] = args.limit
    return Config.from_env(**overrides)


def cmd_config(args) -> int:
    config = _load(args)
    for key, value in config.describe().items():
        print(f"{key:32} {value}")
    return 0


def cmd_doctor(args) -> int:
    from .brokers import BrokerError, LiveBroker
    from .data import DataError

    config = _load(args)
    setup_logging("INFO", config.log_file, config.log_json)
    ok = True

    print(f"polymarket-bot {__version__}")
    print(f"mode                 {config.mode}")
    print(f"strategies           arb={config.arb_enabled} basket={config.basket_enabled} fade={config.fade_enabled}")
    print()

    print("[1/4] configuration")
    try:
        config.validate()
        print("      OK")
    except ConfigError as exc:
        ok = False
        print(f"      FAIL: {exc}")

    print("[2/4] market data connectivity")
    scanner = MarketScanner(config)
    # Bound before the fetch so every later section degrades to a SKIP/WARN
    # when Gamma is unreachable, instead of crashing on an unbound name.
    markets: list = []
    try:
        markets = scanner.fetch_binary_markets(limit=3)
        print(f"      OK: fetched {len(markets)} markets from Gamma")
    except DataError as exc:
        ok = False
        print(f"      FAIL: {exc}")
    try:
        books = scanner.fetch_books([m.token_ids[0] for m in markets[:2]]) if markets else {}
        if books:
            sample = next(iter(books.values()))
            print(
                f"      OK: book has {len(sample.bids)} bids / {len(sample.asks)} asks, "
                f"best bid={sample.best_bid} best ask={sample.best_ask}"
            )
            if sample.asks and sample.asks[0].price > sample.asks[-1].price:
                print("      OK: best ask is the lowest price (book sorted correctly)")
        else:
            print("      WARN: no books returned")
    except DataError as exc:
        ok = False
        print(f"      FAIL: {exc}")

    print("[3/4] fee model")
    if markets:
        info = markets[0]
        rate = scanner.fee_rate_for(info)
        print(
            f"      OK: {info.question[:40]!r} feesEnabled={info.fees_enabled} "
            f"feeType={info.fee_type} taker_rate={rate}"
        )
    else:
        print("      SKIP: no market to inspect")

    print("[4/4] credentials / live trading")
    if config.is_live:
        if not config.has_credentials:
            ok = False
            print("      FAIL: live mode requires POLYMARKET_PRIVATE_KEY")
        else:
            try:
                broker = LiveBroker(config)
                balance = broker.balance_usd()
                print(f"      OK: connected via {broker.backend}; balance=${balance:.2f}")
                broker.close()
            except BrokerError as exc:
                ok = False
                print(f"      FAIL: {exc}")
    else:
        print("      OK: paper mode needs no credentials")

    print()
    print("doctor:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_scan(args) -> int:
    config = _load(args)
    setup_logging(config.log_level, config.log_file, config.log_json)
    scanner = MarketScanner(config)
    from .strategies import StrategyEngine

    engine = StrategyEngine(config)
    groups = scanner.scan()
    found = []
    for group in groups:
        from .fees import FeeModel

        fee = FeeModel.for_market(
            fees_enabled=group.fees_enabled,
            fee_type=group.metadata.get("fee_type"),
            override=config.taker_fee_rate_override or None,
            safety_multiplier=config.fee_safety_multiplier,
            maker=config.assume_maker,
        )
        for sig in engine.evaluate(group, fee):
            found.append((group, sig))

    found.sort(key=lambda item: item[1].edge_per_set, reverse=True)

    if args.json:
        payload = {
            "groups_scanned": len(groups),
            "opportunities": [
                {
                    "kind": sig.kind,
                    "title": sig.title,
                    "edge_per_set": str(sig.edge_per_set),
                    "cost_per_set": str(sig.cost_per_set),
                    "max_sets": str(sig.max_sets),
                    "notional_usd": str(sig.notional_usd),
                    "expected_profit_usd": str(sig.expected_profit_usd),
                    "confidence": sig.confidence,
                    "legs": [
                        {"outcome": l.outcome_name, "price": str(l.price), "shares": str(l.shares)}
                        for l in sig.legs
                    ],
                }
                for _, sig in found
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"scanned {len(groups)} market group(s)")
    if args.show_all:
        for group in groups:
            ask = group.ask_sum()
            print(
                f"  {group.title[:52]:54} n={group.n_outcomes} "
                f"ask_sum={ask if ask is None else f'{ask:.4f}'} vol24h=${group.volume_24h:,.0f}"
            )
        print()

    if not found:
        print("no opportunities above the configured thresholds.")
        print("this is normal: sub-$1 complete sets are rare and short-lived.")
        return 0

    print(f"{len(found)} opportunit(ies):")
    for group, sig in found:
        print(f"  {sig.describe()}")
        # A directional signal's "profit" is its own model's assumption, not a
        # locked-in number. Calling both "expected_profit" would present the
        # guess and the guarantee as the same kind of figure.
        directional = bool(sig.metadata.get("directional")) or bool(
            sig.metadata.get("unverified_relation")
        )
        label = "assumed_profit" if directional else "locked_profit"
        print(
            f"      max_sets={sig.max_sets} {label}=${sig.expected_profit_usd:.2f} "
            f"confidence={sig.confidence:.2f} strategy={STRATEGY_LABELS.get(sig.kind, sig.kind)}"
        )
    return 0


def cmd_ladders(args) -> int:
    """List candidate ladders so an operator can confirm the relation.

    Detection is a guess from question wording. This command exists to make
    that guess inspectable: it prints each pair with the ids needed to
    allowlist it, and whether the resolution question really does imply the
    other one is a judgement the operator has to make by opening both markets.
    """
    config = _load(args)
    setup_logging(config.log_level, config.log_file, config.log_json)
    from .cross_market import candidate_from_market
    from .relations import adjacent_pairs, group_threshold_ladders

    scanner = MarketScanner(config)
    limit = args.limit or config.scan_limit
    # Gamma caps a page at 100 rows, so ask for enough pages to reach the
    # requested depth rather than silently scanning only the top 100.
    pages = max(1, (limit + 99) // 100)
    infos = scanner.fetch_binary_markets(limit=limit, pages=pages)
    candidates = [c for c in (candidate_from_market(i) for i in infos) if c is not None]
    ladders = group_threshold_ladders(candidates)
    confirmed = config.confirmed_cross_market_pairs

    if args.json:
        payload = {
            "markets_scanned": len(infos),
            "candidates": len(candidates),
            "ladders": [
                {
                    "subject": key.split("|")[0],
                    "direction": key.split("|")[1],
                    "members": [
                        {
                            "threshold": str(c.threshold),
                            "market_id": c.market_id,
                            "condition_id": c.condition_id,
                            "question": c.question,
                        }
                        for c in ladder
                    ],
                    "pairs": [
                        {
                            "pair_id": f"{lo.condition_id}:{hi.condition_id}",
                            "confirmed": tuple(sorted((lo.condition_id, hi.condition_id))) in confirmed,
                            "lower": lo.question,
                            "higher": hi.question,
                        }
                        for lo, hi in adjacent_pairs(ladder)
                    ],
                }
                for key, ladder in ladders.items()
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"scanned {len(infos)} market(s), {len(candidates)} with a readable threshold")
    if not ladders:
        print("no threshold ladders found.")
        return 0

    for key, ladder in ladders.items():
        subject, direction = key.rsplit("|", 1)
        print(f"\n{direction.upper()} ladder: {subject}")
        for candidate in ladder:
            print(f"    {str(candidate.threshold).rjust(12)}  {candidate.question[:64]}")
        for lo, hi in adjacent_pairs(ladder):
            pair_id = f"{lo.condition_id}:{hi.condition_id}"
            mark = "confirmed" if tuple(sorted((lo.condition_id, hi.condition_id))) in confirmed else "NOT confirmed"
            print(f"    pair {mark}: {pair_id}")
    print(
        "\nDetection is a guess from wording, not a proof. Open both markets and check "
        "the resolution rules agree before adding a pair to "
        "POLYMARKET_BOT_CROSS_MARKET_CONFIRMED_PAIRS."
    )
    return 0


def cmd_run(args) -> int:
    config = _load(args)
    setup_logging(config.log_level, config.log_file, config.log_json)
    from .engine import TradingEngine

    print("=" * 72)
    print(f"polymarket-bot {__version__}")
    print(f"mode: {config.mode}" + ("  (no real money moves)" if not config.is_live else "  (REAL ORDERS)"))
    print(
        f"strategies: arb={config.arb_enabled} basket={config.basket_enabled} fade={config.fade_enabled}"
        f" cross_market={config.cross_market_enabled}"
        f" | max_order=${config.max_order_usd:.0f} | max_exposure=${config.max_total_exposure_usd:.0f}"
    )
    if config.cross_market_enabled and config.is_live and not config.confirmed_cross_market_pairs:
        print(
            "NOTE: cross-market is on in live mode but no pairs are confirmed. "
            "Run `polymarket-bot ladders` and set "
            "POLYMARKET_BOT_CROSS_MARKET_CONFIRMED_PAIRS; nothing will trade until then."
        )
    if config.is_live and not config.has_credentials:
        print("ERROR: live mode requested without credentials. Set POLYMARKET_PRIVATE_KEY.")
        return 2
    print("=" * 72)

    try:
        engine = TradingEngine(config)
    except Exception as exc:  # noqa: BLE001
        print(f"failed to start: {exc}")
        return 2
    engine.run_forever(max_cycles=args.max_cycles)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    handlers = {
        "run": cmd_run,
        "scan": cmd_scan,
        "doctor": cmd_doctor,
        "config": cmd_config,
        "ladders": cmd_ladders,
    }
    try:
        return handlers[args.command](args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
