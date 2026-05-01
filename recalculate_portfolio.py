"""Recalculate backtest outputs for the current portfolio universe."""

from __future__ import annotations

import json

from backtester import Backtester
from config import SETTINGS
from features import run_features


def main() -> None:
    print("Recalculating backtest for symbols:", ", ".join(SETTINGS.symbols))
    features = run_features()
    metrics = Backtester().run(features)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
