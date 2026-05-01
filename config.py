"""Central configuration for the alpha generation project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    symbols: List[str] = field(
        default_factory=lambda: ["NVDA", "MSFT", "GOOG", "TLT", "CRWV", "NBIS", "BE"]
    )
    start_date: date = date(2020, 1, 1)
    end_date: date = date(2025, 12, 31)

    initial_capital: float = 100_000.0
    slippage_bps: float = 10.0  # 0.10%
    fee_bps: float = 5.0  # 0.05%
    max_portfolio_drawdown: float = 0.10
    stop_loss_pct: float = 0.02
    max_position_pct: float = 0.25
    base_position_pct: float = 0.08

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    db_path: Path = Path("data/market_data.sqlite")
    features_path: Path = Path("data/features.csv")
    equity_curve_path: Path = Path("outputs/equity_curve.csv")
    metrics_path: Path = Path("outputs/metrics.json")
    trades_path: Path = Path("outputs/trades.csv")

    alpaca_api_key: str | None = os.getenv("ALPACA_API_KEY")
    alpaca_secret_key: str | None = os.getenv("ALPACA_SECRET_KEY")
    fred_api_key: str | None = os.getenv("FRED_API_KEY")
    news_api_key: str | None = os.getenv("NEWS_API_KEY")

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 10_000.0

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
