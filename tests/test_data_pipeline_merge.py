import pandas as pd

from data_pipeline import DataPipeline


def test_merge_all_handles_missing_sentiment_symbol_key() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "symbol": ["NVDA", "NVDA"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1_000_000, 1_100_000],
            "vwap": [100.3, 101.2],
            "trade_count": [1500, 1600],
        }
    )
    macro = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "dgs10": [3.8, 3.7],
            "dgs2": [3.5, 3.4],
            "cpi": [280.0, 280.1],
            "fedfunds": [4.5, 4.5],
            "vix": [18.0, 19.0],
        }
    )
    # Intentionally omit symbol to test merge guardrails.
    sentiment = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "sentiment_score": [0.6, 0.55],
        }
    )

    merged = DataPipeline.merge_all(market, macro, sentiment)
    assert not merged.empty
    assert {"date", "symbol", "sentiment_score", "vix"}.issubset(merged.columns)
    assert merged["sentiment_score"].notna().all()


def test_merge_all_recovers_symbol_from_symbol_x() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "symbol_x": ["MSFT", "MSFT"],
            "open": [200.0, 201.0],
            "high": [201.0, 202.0],
            "low": [199.0, 200.0],
            "close": [200.5, 201.5],
            "volume": [900_000, 910_000],
            "vwap": [200.2, 201.2],
            "trade_count": [1200, 1250],
        }
    )
    macro = pd.DataFrame({"date": pd.to_datetime(["2025-01-02", "2025-01-03"]), "vix": [20.0, 21.0]})
    sentiment = pd.DataFrame({"date": pd.to_datetime(["2025-01-02", "2025-01-03"]), "sentiment_score": [0.5, 0.5]})

    merged = DataPipeline.merge_all(market, macro, sentiment)
    assert "symbol" in merged.columns
    assert (merged["symbol"] == "MSFT").all()
