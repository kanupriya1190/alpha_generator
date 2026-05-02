import pandas as pd

from data_pipeline import DataPipeline
from features import run_features


def _fake_market(symbols, start, end) -> pd.DataFrame:
    dates = pd.date_range(start="2025-01-01", periods=70, freq="B")
    rows = []
    for symbol in symbols:
        for i, d in enumerate(dates):
            px = 100 + i * 0.5
            rows.append(
                {
                    "date": d,
                    "symbol": symbol,
                    "open": px - 0.2,
                    "high": px + 0.6,
                    "low": px - 0.6,
                    "close": px,
                    "volume": 1_000_000 + i * 1000,
                    "vwap": px,
                    "trade_count": 1000 + i,
                }
            )
    return pd.DataFrame(rows)


def _fake_macro(start, end) -> pd.DataFrame:
    dates = pd.date_range(start="2025-01-01", periods=70, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "dgs10": 3.8,
            "dgs2": 3.5,
            "cpi": 280.0,
            "fedfunds": 4.5,
            "vix": 18.0,
        }
    )


def _fake_sentiment(symbols, start, end) -> pd.DataFrame:
    dates = pd.date_range(start="2025-01-01", periods=70, freq="B")
    rows = []
    for symbol in symbols:
        for d in dates:
            rows.append({"date": d, "symbol": symbol, "sentiment_score": 0.55})
    return pd.DataFrame(rows)


def test_run_features_produces_expected_columns(monkeypatch) -> None:
    monkeypatch.setattr(DataPipeline, "fetch_market_data", lambda self, symbols, start, end: _fake_market(symbols, start, end))
    monkeypatch.setattr(DataPipeline, "fetch_macro_data", lambda self, start, end: _fake_macro(start, end))
    monkeypatch.setattr(DataPipeline, "fetch_sentiment_data", lambda self, symbols, start, end: _fake_sentiment(symbols, start, end))
    monkeypatch.setattr(DataPipeline, "save_sqlite", lambda self, market, macro, sentiment: None)

    out = run_features()
    assert not out.empty
    assert {"symbol", "date", "momentum_10d", "rsi_14", "sentiment_momentum", "yield_curve_slope"}.issubset(out.columns)
    assert out["symbol"].nunique() >= 1
