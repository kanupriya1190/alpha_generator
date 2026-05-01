"""CLI entrypoint for one-cycle Alpaca paper trading."""

from __future__ import annotations

import argparse
import json

from live_trader import LiveTrader


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Alpaca paper-trading cycle.")
    parser.add_argument("--execute", action="store_true", help="Submit live paper orders (default is dry-run).")
    parser.add_argument("--no-stats-arb", action="store_true", help="Disable stats-arbitrage weight adjustments.")
    parser.add_argument("--no-rebalance", action="store_true", help="Disable target-weight rebalancing.")
    args = parser.parse_args()

    out = LiveTrader().run_once(
        dry_run=not args.execute,
        use_stats_arb=not args.no_stats_arb,
        rebalance=not args.no_rebalance,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
