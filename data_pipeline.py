"""Data ingestion and storage for multi-agent alpha generation."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf
try:
    from pandas_datareader import data as pdr
except Exception as exc:  # pragma: no cover - dependency/runtime path
    pdr = None
    print(f"[WARN] pandas_datareader unavailable; using macro fallback path: {exc}")

from config import SETTINGS


def _to_utc_timestamp(date_like: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(date_like)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass
class DataPipeline:
    """Fetches market, macro, and sentiment data and stores artifacts."""

    def fetch_market_data(self, symbols: Iterable[str], start: datetime, end: datetime) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for symbol in symbols:
            frame = self._fetch_market_for_symbol(symbol, start, end)
            frames.append(frame)
        market = pd.concat(frames, axis=0, ignore_index=True)
        market["date"] = pd.to_datetime(market["date"], utc=True).dt.tz_localize(None)
        market = market.sort_values(["symbol", "date"]).reset_index(drop=True)
        return market

    def _fetch_market_for_symbol(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        if SETTINGS.alpaca_api_key and SETTINGS.alpaca_secret_key:
            try:
                return self._fetch_market_alpaca(symbol, start, end)
            except Exception as exc:  # pragma: no cover - network/runtime path
                print(f"[WARN] Alpaca fetch failed for {symbol}: {exc}. Falling back to yfinance.")
        try:
            return self._fetch_market_yfinance(symbol, start, end)
        except Exception as exc:  # pragma: no cover - network/runtime path
            print(f"[WARN] yfinance fetch failed for {symbol}: {exc}. Using synthetic market data.")
            return self._generate_synthetic_market_data(symbol, start, end)

    def _fetch_market_alpaca(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        headers = {
            "APCA-API-KEY-ID": SETTINGS.alpaca_api_key or "",
            "APCA-API-SECRET-KEY": SETTINGS.alpaca_secret_key or "",
        }
        params = {
            "symbols": symbol,
            "timeframe": "1Day",
            "start": _to_utc_timestamp(start).isoformat(),
            "end": _to_utc_timestamp(end).isoformat(),
            "adjustment": "all",
            "limit": 10_000,
            "feed": "iex",
        }
        url = "https://data.alpaca.markets/v2/stocks/bars"
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        bars = payload.get("bars", {}).get(symbol, [])
        if not bars:
            raise ValueError(f"No bars returned from Alpaca for {symbol}")

        df = pd.DataFrame(bars)
        df = df.rename(
            columns={
                "t": "date",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "trade_count",
            }
        )
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["symbol"] = symbol
        return df[["date", "symbol", "open", "high", "low", "close", "volume", "vwap", "trade_count"]]

    def _fetch_market_yfinance(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        raw = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        if raw.empty:
            raise ValueError(f"No data from yfinance for {symbol}")

        # yfinance may return MultiIndex columns: (Price, Ticker). Normalize to flat columns.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]

        raw = raw.rename(
            columns={
                "adj close": "adj_close",
            }
        )
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(raw.columns)):
            raise ValueError(f"Unexpected yfinance columns for {symbol}: {list(raw.columns)}")
        raw["vwap"] = np.nan
        raw["trade_count"] = np.nan
        raw["symbol"] = symbol
        raw = raw.reset_index().rename(columns={"Date": "date", "date": "date"})
        return raw[["date", "symbol", "open", "high", "low", "close", "volume", "vwap", "trade_count"]]

    def _generate_synthetic_market_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        dates = pd.date_range(start=start, end=end, freq="B")
        if len(dates) == 0:
            raise ValueError("No business dates for synthetic data generation")
        seed = abs(hash(symbol)) % (2**32)
        rng = np.random.default_rng(seed)
        drift = 0.0003
        vol = 0.015
        returns = rng.normal(drift, vol, len(dates))
        close = 100 * np.cumprod(1 + returns)
        open_px = close * (1 + rng.normal(0, 0.002, len(dates)))
        high = np.maximum(open_px, close) * (1 + np.abs(rng.normal(0, 0.003, len(dates))))
        low = np.minimum(open_px, close) * (1 - np.abs(rng.normal(0, 0.003, len(dates))))
        volume = rng.integers(500_000, 5_000_000, len(dates))
        out = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": open_px,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "vwap": np.nan,
                "trade_count": np.nan,
            }
        )
        return out

    def fetch_macro_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        # FRED series IDs: DGS10, DGS2, CPIAUCSL, FEDFUNDS.
        series_map: Dict[str, str] = {
            "dgs10": "DGS10",
            "dgs2": "DGS2",
            "cpi": "CPIAUCSL",
            "fedfunds": "FEDFUNDS",
        }
        macro_frames = []
        for alias, fred_id in series_map.items():
            try:
                if pdr is None:
                    raise RuntimeError("pandas_datareader import failed")
                s = pdr.DataReader(fred_id, "fred", start, end).rename(columns={fred_id: alias})
                macro_frames.append(s)
            except Exception as exc:  # pragma: no cover - network/runtime path
                print(f"[WARN] FRED fetch failed for {fred_id}: {exc}")

        if macro_frames:
            macro = pd.concat(macro_frames, axis=1)
        else:
            # Last-resort fallback to keep pipeline runnable.
            date_index = pd.date_range(start=start, end=end, freq="B")
            macro = pd.DataFrame(index=date_index)
            macro["dgs10"] = 3.0
            macro["dgs2"] = 2.0
            macro["cpi"] = np.linspace(250, 300, len(macro))
            macro["fedfunds"] = 2.5

        vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
        if not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = [str(c[0]).lower() for c in vix.columns]
            else:
                vix.columns = [str(c).lower() for c in vix.columns]
            if "close" in vix.columns:
                macro["vix"] = vix["close"]
            else:
                macro["vix"] = 20.0
        else:
            macro["vix"] = 20.0

        macro = macro.sort_index().ffill().bfill()
        macro = macro.reset_index()
        if "DATE" in macro.columns:
            macro = macro.rename(columns={"DATE": "date"})
        elif "Date" in macro.columns:
            macro = macro.rename(columns={"Date": "date"})
        elif "index" in macro.columns:
            macro = macro.rename(columns={"index": "date"})
        macro["date"] = pd.to_datetime(macro["date"])
        return macro

    def fetch_sentiment_data(self, symbols: Iterable[str], start: datetime, end: datetime) -> pd.DataFrame:
        # Default approach: deterministic proxy sentiment from short-term returns,
        # keeping the project runnable without paid news endpoints.
        market = self.fetch_market_data(symbols=symbols, start=start, end=end)
        market = market.sort_values(["symbol", "date"])
        market["ret_3d"] = market.groupby("symbol")["close"].pct_change(3)
        market["ret_10d"] = market.groupby("symbol")["close"].pct_change(10)
        proxy = (0.5 * market["ret_3d"].fillna(0) + 0.5 * market["ret_10d"].fillna(0)).clip(-0.05, 0.05)
        market["sentiment_score"] = ((proxy + 0.05) / 0.10).clip(0, 1)
        return market[["date", "symbol", "sentiment_score"]]

    @staticmethod
    def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
        clean = df.copy()
        clean = clean.drop_duplicates(subset=["symbol", "date"]).sort_values(["symbol", "date"])
        for col in ["open", "high", "low", "close", "volume"]:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean = clean.groupby("symbol", group_keys=False).apply(lambda x: x.ffill().bfill())
        clean = clean[clean["close"] > 0]
        return clean.reset_index(drop=True)

    @staticmethod
    def merge_all(market: pd.DataFrame, macro: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
        market = market.copy()
        macro = macro.copy()
        sentiment = sentiment.copy()

        if "symbol" not in market.columns:
            market = market.reset_index()
        if "symbol" not in market.columns and "symbol_x" in market.columns:
            market["symbol"] = market["symbol_x"]

        # Streamlit Cloud/runtime variants can produce missing columns; repair shape defensively.
        if "date" not in macro.columns:
            macro = pd.DataFrame({"date": market["date"].drop_duplicates().sort_values()})
        if "symbol" in macro.columns:
            macro = macro.drop(columns=["symbol"])
        if "date" not in sentiment.columns:
            sentiment = market[["date", "symbol"]].copy()
            sentiment["sentiment_score"] = 0.5
        if "symbol" not in sentiment.columns and "symbol_x" in sentiment.columns:
            sentiment["symbol"] = sentiment["symbol_x"]
        if "symbol" not in sentiment.columns and "symbol_y" in sentiment.columns:
            sentiment["symbol"] = sentiment["symbol_y"]
        if "symbol" not in sentiment.columns:
            sentiment = sentiment.merge(
                market[["date", "symbol"]].drop_duplicates(), on="date", how="left"
            )
        if "sentiment_score" not in sentiment.columns:
            sentiment["sentiment_score"] = 0.5
        sentiment = sentiment[["date", "symbol", "sentiment_score"]].drop_duplicates(["date", "symbol"])

        market["date"] = pd.to_datetime(market["date"]).dt.normalize()
        macro["date"] = pd.to_datetime(macro["date"]).dt.normalize()
        sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.normalize()

        merged = market.merge(macro, on="date", how="left")
        if "symbol" not in merged.columns and "symbol_x" in merged.columns:
            merged["symbol"] = merged["symbol_x"]
        if "symbol" not in merged.columns and "symbol_y" in merged.columns:
            merged["symbol"] = merged["symbol_y"]
        # Final guardrail for Cloud runtime key-shape quirks: never crash on sentiment merge.
        if "date" not in merged.columns and "date_x" in merged.columns:
            merged["date"] = merged["date_x"]
        if "date" not in merged.columns and "date_y" in merged.columns:
            merged["date"] = merged["date_y"]

        if {"date", "symbol"}.issubset(merged.columns) and {"date", "symbol"}.issubset(sentiment.columns):
            try:
                merged = merged.merge(sentiment, on=["date", "symbol"], how="left")
            except KeyError as exc:
                print(f"[WARN] Sentiment merge key error ({exc}); falling back to date-only sentiment merge.")
                date_sent = (
                    sentiment.groupby("date", as_index=False)["sentiment_score"].mean()
                    if "date" in sentiment.columns and "sentiment_score" in sentiment.columns
                    else pd.DataFrame(columns=["date", "sentiment_score"])
                )
                merged = merged.merge(date_sent, on="date", how="left")
        else:
            print(
                "[WARN] Skipping sentiment merge due to missing keys. "
                f"merged_cols={list(merged.columns)} sentiment_cols={list(sentiment.columns)}"
            )
            merged["sentiment_score"] = 0.5
        for col, fallback in {"dgs10": 3.0, "dgs2": 2.0, "cpi": 275.0, "fedfunds": 2.5, "vix": 20.0}.items():
            if col not in merged.columns:
                merged[col] = fallback
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(fallback)
        merged["sentiment_score"] = merged["sentiment_score"].fillna(0.5)
        merged = merged.sort_values(["symbol", "date"]).ffill().bfill()
        return merged

    def save_sqlite(self, market: pd.DataFrame, macro: pd.DataFrame, sentiment: pd.DataFrame) -> None:
        SETTINGS.ensure_dirs()
        conn = sqlite3.connect(SETTINGS.db_path)
        market.to_sql("market_data", conn, if_exists="replace", index=False)
        macro.to_sql("macro_data", conn, if_exists="replace", index=False)
        sentiment.to_sql("sentiment_data", conn, if_exists="replace", index=False)
        conn.close()

    @staticmethod
    def validate_dataset(df: pd.DataFrame) -> None:
        required = {"date", "symbol", "open", "high", "low", "close", "volume", "vix", "sentiment_score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
        if df["date"].isna().any():
            raise ValueError("Missing dates in merged dataset")
        if (df["close"] <= 0).any():
            raise ValueError("Non-positive close prices found")

    def run(self) -> pd.DataFrame:
        SETTINGS.ensure_dirs()
        start = datetime.combine(SETTINGS.start_date, datetime.min.time())
        end = datetime.combine(SETTINGS.end_date, datetime.min.time())

        market = self.fetch_market_data(SETTINGS.symbols, start, end)
        market = self.clean_market_data(market)
        macro = self.fetch_macro_data(start, end)
        sentiment = self.fetch_sentiment_data(SETTINGS.symbols, start, end)

        self.save_sqlite(market, macro, sentiment)
        merged = self.merge_all(market, macro, sentiment)
        self.validate_dataset(merged)
        merged.to_csv("data/raw_merged.csv", index=False)
        return merged


def main() -> None:
    pipeline = DataPipeline()
    df = pipeline.run()
    print(f"Data pipeline complete. Rows: {len(df)}, symbols: {df['symbol'].nunique()}")


if __name__ == "__main__":
    main()
