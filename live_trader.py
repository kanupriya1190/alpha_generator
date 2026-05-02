"""Live paper-trading integration for Alpaca (no MCP dependency)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
try:
    from pandas_datareader import data as pdr
except Exception as exc:  # pragma: no cover - dependency/runtime path
    pdr = None
    print(f"[WARN] pandas_datareader unavailable in live_trader; using macro fallback path: {exc}")

from config import SETTINGS
from news_sentiment import FinBERTNewsSentiment
from orchestrator import Orchestrator


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _download_close_history(symbols: List[str], lookback_days: int = 180) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    closes: Dict[str, pd.Series] = {}
    for sym in symbols:
        raw = yf.download(sym, start=start, end=end, auto_adjust=False, progress=False)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]
        if "close" not in raw.columns:
            continue
        closes[sym] = raw["close"].dropna()
    if not closes:
        return pd.DataFrame()
    out = pd.DataFrame(closes).dropna(how="all").ffill().bfill()
    return out


def _stats_arb_adjustments(
    close_history: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    entry_z: float = 1.5,
    weight_shift: float = 0.03,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Generate additive portfolio weight tweaks from pair spread z-scores."""
    adjustments: Dict[str, float] = {}
    diagnostics: List[Dict[str, Any]] = []
    if close_history.empty:
        return adjustments, diagnostics

    for a, b in pairs:
        if a not in close_history.columns or b not in close_history.columns:
            continue
        pair_df = close_history[[a, b]].dropna().tail(120)
        if len(pair_df) < 40:
            continue

        spread = np.log(pair_df[a]) - np.log(pair_df[b])
        mean = spread.rolling(60).mean().iloc[-1]
        std = spread.rolling(60).std().iloc[-1]
        if pd.isna(mean) or pd.isna(std) or std < 1e-8:
            continue

        z = float((spread.iloc[-1] - mean) / std)
        regime = "neutral"
        if z > entry_z:
            # a appears rich vs b -> rotate weight from a to b
            adjustments[a] = adjustments.get(a, 0.0) - weight_shift
            adjustments[b] = adjustments.get(b, 0.0) + weight_shift
            regime = f"{a}_rich"
        elif z < -entry_z:
            # b appears rich vs a -> rotate weight from b to a
            adjustments[b] = adjustments.get(b, 0.0) - weight_shift
            adjustments[a] = adjustments.get(a, 0.0) + weight_shift
            regime = f"{b}_rich"

        diagnostics.append(
            {
                "pair": f"{a}/{b}",
                "zscore": z,
                "entry_threshold": entry_z,
                "weight_shift": weight_shift if regime != "neutral" else 0.0,
                "regime": regime,
            }
        )

    return adjustments, diagnostics


def _normalize_weights(weights: Dict[str, float], max_total: float = 0.80) -> Dict[str, float]:
    clipped = {k: float(np.clip(v, 0.0, max_total)) for k, v in weights.items()}
    gross = sum(clipped.values())
    if gross <= max_total or gross <= 1e-9:
        return clipped
    scale = max_total / gross
    return {k: v * scale for k, v in clipped.items()}


def _fetch_live_macro_features() -> Dict[str, float]:
    """Fetch latest rates data; fallback gracefully if unavailable."""
    out = {
        "dgs10": 4.0,
        "dgs2": 3.8,
        "fedfunds": 4.5,
        "yield_curve_slope": 0.2,
        "yield_10y_change_21d": 0.0,
    }
    end = datetime.utcnow()
    start = end - timedelta(days=90)
    try:
        if pdr is None:
            raise RuntimeError("pandas_datareader import failed")
        dgs10 = pdr.DataReader("DGS10", "fred", start, end)["DGS10"].dropna()
        dgs2 = pdr.DataReader("DGS2", "fred", start, end)["DGS2"].dropna()
        fed = pdr.DataReader("FEDFUNDS", "fred", start, end)["FEDFUNDS"].dropna()
        if not dgs10.empty:
            out["dgs10"] = float(dgs10.iloc[-1])
            if len(dgs10) > 21:
                out["yield_10y_change_21d"] = float(dgs10.iloc[-1] - dgs10.iloc[-21])
        if not dgs2.empty:
            out["dgs2"] = float(dgs2.iloc[-1])
        if not fed.empty:
            out["fedfunds"] = float(fed.iloc[-1])
        out["yield_curve_slope"] = out["dgs10"] - out["dgs2"]
    except Exception:
        pass
    return out


NEWS_SENTIMENT = FinBERTNewsSentiment()


@dataclass
class AlpacaPaperClient:
    """Minimal Alpaca REST client for paper trading."""

    base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    data_url: str = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

    def __post_init__(self) -> None:
        self.key = SETTINGS.alpaca_api_key or ""
        self.secret = SETTINGS.alpaca_secret_key or ""

    @property
    def enabled(self) -> bool:
        return bool(self.key and self.secret)

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, payload: Dict[str, Any] | None = None) -> Any:
        if not self.enabled:
            raise ValueError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY in environment.")
        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, headers=self.headers, json=payload, timeout=30)
        resp.raise_for_status()
        if not resp.text:
            return {}
        return resp.json()

    def get_account(self) -> Dict[str, Any]:
        return self._request("GET", "/v2/account")

    def list_positions(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/v2/positions")

    def list_orders(self, status: str = "all", limit: int = 50) -> List[Dict[str, Any]]:
        return self._request("GET", f"/v2/orders?status={status}&limit={limit}&direction=desc")

    def place_market_order(self, symbol: str, qty: int, side: str) -> Dict[str, Any]:
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        return self._request("POST", "/v2/orders", payload=payload)


def _build_signal_row(symbol: str) -> pd.Series:
    """Compute latest row with required features for orchestrator."""
    end = datetime.utcnow()
    start = end - timedelta(days=180)
    raw = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"No recent market data for {symbol}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in raw.columns:
            raise ValueError(f"Missing {col} for {symbol}")

    raw = raw.dropna(subset=["close"]).copy()
    raw["ret_1d"] = raw["close"].pct_change(1)
    raw["momentum_10d"] = raw["close"].pct_change(10)
    raw["momentum_20d"] = raw["close"].pct_change(20)
    raw["ma_20"] = raw["close"].rolling(20).mean()
    raw["std_20"] = raw["close"].rolling(20).std()
    raw["zscore_20"] = (raw["close"] - raw["ma_20"]) / raw["std_20"]
    raw["volume_ma_20"] = raw["volume"].rolling(20).mean()
    raw["volume_ratio"] = raw["volume"] / raw["volume_ma_20"]
    raw["price_vs_ma20"] = raw["close"] / raw["ma_20"] - 1

    # Sentiment proxy for always-runnable signal generation.
    raw["ret_3d"] = raw["close"].pct_change(3)
    raw["ret_10d"] = raw["close"].pct_change(10)
    proxy = (0.5 * raw["ret_3d"].fillna(0) + 0.5 * raw["ret_10d"].fillna(0)).clip(-0.05, 0.05)
    raw["sentiment_score"] = ((proxy + 0.05) / 0.10).clip(0, 1)
    raw["sentiment_momentum"] = raw["sentiment_score"].rolling(3).mean() - raw["sentiment_score"].rolling(10).mean()
    proxy_sent = _safe_float(raw["sentiment_score"].iloc[-1], 0.5)
    news_details = NEWS_SENTIMENT.score_symbol_details(symbol)
    news_sent = _safe_float(news_details.get("score"), 0.5)
    headline_count = int(_safe_float(news_details.get("headline_count"), 0))
    sentiment_source = str(news_details.get("source", "unknown"))
    # Blend model-based news sentiment with proxy sentiment.
    blended_sent = 0.6 * news_sent + 0.4 * proxy_sent

    # Macro proxy for live execution path.
    vix = 20.0
    try:
        vix_hist = yf.download("^VIX", start=end - timedelta(days=14), end=end, auto_adjust=False, progress=False)
        if not vix_hist.empty:
            if isinstance(vix_hist.columns, pd.MultiIndex):
                vix_hist.columns = [c[0].lower() for c in vix_hist.columns]
            else:
                vix_hist.columns = [str(c).lower() for c in vix_hist.columns]
            vix = float(vix_hist["close"].dropna().iloc[-1])
    except Exception:
        vix = 20.0

    latest = raw.iloc[-1]
    macro = _fetch_live_macro_features()
    return pd.Series(
        {
            "symbol": symbol,
            "close": _safe_float(latest.get("close")),
            "momentum_10d": _safe_float(latest.get("momentum_10d")),
            "momentum_20d": _safe_float(latest.get("momentum_20d")),
            "zscore_20": _safe_float(latest.get("zscore_20")),
            "volume_ratio": _safe_float(latest.get("volume_ratio"), 1.0),
            "sentiment_score": blended_sent,
            "sentiment_momentum": _safe_float(latest.get("sentiment_momentum"), 0.0),
            "news_sentiment_score": news_sent,
            "news_headline_count": headline_count,
            "sentiment_source": sentiment_source,
            "vix": vix,
            "macro_headwinds": 1.0 if vix > 25 else 0.0,
            "dgs10": macro["dgs10"],
            "dgs2": macro["dgs2"],
            "fedfunds": macro["fedfunds"],
            "yield_curve_slope": macro["yield_curve_slope"],
            "yield_10y_change_21d": macro["yield_10y_change_21d"],
            "price_vs_ma20": _safe_float(latest.get("price_vs_ma20")),
        }
    )


@dataclass
class LiveTrader:
    """Run one live paper-trading cycle based on orchestrated signals."""

    orchestrator: Orchestrator = field(default_factory=Orchestrator)
    client: AlpacaPaperClient = field(default_factory=AlpacaPaperClient)

    def run_once(self, dry_run: bool = True, use_stats_arb: bool = True, rebalance: bool = True) -> Dict[str, Any]:
        account = self.client.get_account()
        equity = _safe_float(account.get("equity"), SETTINGS.initial_capital)

        positions = self.client.list_positions()
        current_qty = {p.get("symbol"): int(float(p.get("qty", 0))) for p in positions if p.get("symbol")}
        current_mv = {p.get("symbol"): _safe_float(p.get("market_value", 0)) for p in positions if p.get("symbol")}

        decisions: Dict[str, Dict[str, Any]] = {}
        feature_rows: Dict[str, pd.Series] = {}
        market_prices: Dict[str, float] = {}
        target_weights: Dict[str, float] = {}
        actions: List[Dict[str, Any]] = []

        for symbol in SETTINGS.symbols:
            row = _build_signal_row(symbol)
            decision = self.orchestrator.combine(row, current_drawdown=0.0)
            price = _safe_float(row["close"])
            market_prices[symbol] = price
            decisions[symbol] = decision
            feature_rows[symbol] = row

            signal = str(decision["signal"])
            suggested = float(decision["position_size"])
            current_weight = current_mv.get(symbol, 0.0) / max(equity, 1e-9)

            if signal == "BUY":
                target_weights[symbol] = max(current_weight, suggested)
            elif signal == "SELL":
                target_weights[symbol] = 0.0
            else:
                # HOLD keeps current exposure; rebalance can still trim based on stats-arb.
                target_weights[symbol] = current_weight

        stats_arb_diag: List[Dict[str, Any]] = []
        if use_stats_arb:
            pairs = [("NVDA", "CRWV"), ("GOOG", "MSFT"), ("NBIS", "BE"), ("TLT", "MSFT")]
            close_hist = _download_close_history(SETTINGS.symbols)
            adj, stats_arb_diag = _stats_arb_adjustments(close_hist, pairs=pairs)
            for sym, delta in adj.items():
                target_weights[sym] = target_weights.get(sym, 0.0) + delta

        # Respect hard directional signals after pair-adjustments.
        for symbol in SETTINGS.symbols:
            signal = str(decisions[symbol]["signal"])
            current_weight = current_mv.get(symbol, 0.0) / max(equity, 1e-9)
            if signal == "SELL":
                target_weights[symbol] = 0.0
            elif signal == "HOLD":
                # HOLD should not force fresh risk-on exposure.
                target_weights[symbol] = min(target_weights.get(symbol, 0.0), current_weight)

        target_weights = _normalize_weights(target_weights, max_total=0.80)

        for symbol in SETTINGS.symbols:
            decision = decisions[symbol]
            signal = str(decision["signal"])
            price = market_prices.get(symbol, 0.0)
            held_qty = current_qty.get(symbol, 0)
            desired_weight = target_weights.get(symbol, 0.0)
            current_weight = current_mv.get(symbol, 0.0) / max(equity, 1e-9)
            if signal == "SELL":
                desired_weight = 0.0
            elif signal == "HOLD":
                desired_weight = min(desired_weight, current_weight)
            target_notional = equity * desired_weight
            target_qty = int(math.floor(target_notional / max(price, 1e-9))) if price > 0 else 0

            delta_qty = target_qty - held_qty
            side = None
            qty = 0
            if rebalance:
                if delta_qty > 0:
                    side = "buy"
                    qty = int(delta_qty)
                elif delta_qty < 0:
                    side = "sell"
                    qty = int(abs(delta_qty))
            else:
                if signal == "BUY" and target_qty > held_qty:
                    side = "buy"
                    qty = int(target_qty - held_qty)
                elif signal == "SELL" and held_qty > 0:
                    side = "sell"
                    qty = int(held_qty)

            order_result = None
            if side and qty > 0 and not dry_run:
                order_result = self.client.place_market_order(symbol=symbol, qty=qty, side=side)

            actions.append(
                {
                    "symbol": symbol,
                    "price": price,
                    "signal": signal,
                    "confidence": float(decision["confidence"]),
                    "news_sentiment_score": _safe_float(feature_rows[symbol].get("news_sentiment_score"), 0.5),
                    "news_headline_count": int(_safe_float(feature_rows[symbol].get("news_headline_count"), 0)),
                    "sentiment_source": str(feature_rows[symbol].get("sentiment_source", "unknown")),
                    "position_size_pct": 100 * float(decision["position_size"]),  # raw orchestrator suggestion
                    "target_weight_pct": 100 * desired_weight,  # post rebalancing + stats-arb
                    "held_qty": held_qty,
                    "target_qty": target_qty,
                    "delta_qty": delta_qty,
                    "order_side": side,
                    "order_qty": qty,
                    "order_result": order_result,
                }
            )

        return {
            "dry_run": dry_run,
            "account": account,
            "positions": positions,
            "stats_arb": stats_arb_diag,
            "target_weights": {k: round(100 * v, 4) for k, v in target_weights.items()},
            "actions": actions,
        }

