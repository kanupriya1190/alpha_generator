"""Run one live paper-trading cycle and persist a JSON run log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from generate_public_snapshot import main as generate_snapshot_main
from live_trader import LiveTrader


def main() -> None:
    out_dir = Path("outputs/live_runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = LiveTrader().run_once(dry_run=False, use_stats_arb=True, rebalance=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"run_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved live run output to: {out_path}")
    generate_snapshot_main()
    print("Updated docs/live_snapshot.json")


if __name__ == "__main__":
    main()
