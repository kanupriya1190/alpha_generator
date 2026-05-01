"""Feature engineering for multi-agent alpha generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import SETTINGS
from data_pipeline import DataPipeline


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)

    grouped = out.groupby("symbol", group_keys=False)
    out["ret_1d"] = grouped["close"].pct_change(1)
    out["ret_5d"] = grouped["close"].pct_change(5)
    out["ret_10d"] = grouped["close"].pct_change(10)
    out["ret_20d"] = grouped["close"].pct_change(20)
    out["ret_50d"] = grouped["close"].pct_change(50)
    out["momentum_10d"] = out["ret_10d"]
    out["momentum_20d"] = out["ret_20d"]
    out["momentum_50d"] = out["ret_50d"]

    out["ma_10"] = grouped["close"].transform(lambda x: x.rolling(10).mean())
    out["ma_20"] = grouped["close"].transform(lambda x: x.rolling(20).mean())
    out["ma_50"] = grouped["close"].transform(lambda x: x.rolling(50).mean())
    out["ma_200"] = grouped["close"].transform(lambda x: x.rolling(200).mean())
    out["price_vs_ma20"] = out["close"] / out["ma_20"] - 1.0
    out["price_vs_ma50"] = out["close"] / out["ma_50"] - 1.0

    out["vol_10d"] = grouped["ret_1d"].transform(lambda x: x.rolling(10).std())
    out["vol_20d"] = grouped["ret_1d"].transform(lambda x: x.rolling(20).std())
    out["vol_50d"] = grouped["ret_1d"].transform(lambda x: x.rolling(50).std())

    out["rsi_14"] = grouped["close"].transform(lambda x: _rsi(x, 14))
    out["ema_12"] = grouped["close"].transform(lambda x: _ema(x, 12))
    out["ema_26"] = grouped["close"].transform(lambda x: _ema(x, 26))
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = grouped["macd"].transform(lambda x: _ema(x, 9))
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["std_20"] = grouped["close"].transform(lambda x: x.rolling(20).std())
    out["bollinger_upper"] = out["ma_20"] + 2 * out["std_20"]
    out["bollinger_lower"] = out["ma_20"] - 2 * out["std_20"]
    out["zscore_20"] = (out["close"] - out["ma_20"]) / out["std_20"]

    out["atr_14"] = grouped.apply(lambda g: _atr(g, 14)).reset_index(level=0, drop=True)
    out["volume_ma_20"] = grouped["volume"].transform(lambda x: x.rolling(20).mean())
    out["volume_ratio"] = out["volume"] / out["volume_ma_20"]

    out["yield_curve_slope"] = out["dgs10"] - out["dgs2"]
    out["yield_10y_change_21d"] = out.groupby("symbol")["dgs10"].diff(21)
    out["inflation_1m_change"] = out.groupby("symbol")["cpi"].pct_change(21)
    out["fedfunds_change_1m"] = out.groupby("symbol")["fedfunds"].diff(21)
    out["vix_change_5d"] = out.groupby("symbol")["vix"].diff(5)
    out["risk_on"] = (out["vix"] < 20).astype(float)
    out["macro_headwinds"] = ((out["vix"] > 25) | (out["yield_curve_slope"] < 0)).astype(float)

    out["sentiment_3d_mean"] = grouped["sentiment_score"].transform(lambda x: x.rolling(3).mean())
    out["sentiment_5d_mean"] = grouped["sentiment_score"].transform(lambda x: x.rolling(5).mean())
    out["sentiment_10d_mean"] = grouped["sentiment_score"].transform(lambda x: x.rolling(10).mean())
    out["sentiment_momentum"] = out["sentiment_3d_mean"] - out["sentiment_10d_mean"]

    feature_cols = [
        c
        for c in out.columns
        if c
        not in {
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "trade_count",
        }
    ]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    out[feature_cols] = out.groupby("symbol")[feature_cols].transform(lambda x: x.ffill().bfill())
    out[feature_cols] = out[feature_cols].fillna(0)

    return out


def run_features() -> pd.DataFrame:
    pipeline = DataPipeline()
    merged = pipeline.run()
    features = engineer_features(merged)
    SETTINGS.ensure_dirs()
    features.to_csv(SETTINGS.features_path, index=False)
    return features


def main() -> None:
    features = run_features()
    print(f"Feature engineering complete. Rows: {len(features)}, columns: {len(features.columns)}")


if __name__ == "__main__":
    main()
