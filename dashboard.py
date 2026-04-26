"""Streamlit dashboard for alpha generation project."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from backtester import Backtester
from features import run_features
from orchestrator import Orchestrator

st.set_page_config(page_title="Multi-Agent Alpha Generator", layout="wide")
st.title("Multi-Agent Alpha Generation Dashboard")


def _load_artifact(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _load_metrics(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Backtest Results", "Signal Performance", "Agent Comparison", "Live Predictions", "Replay"]
)

equity_curve = _load_artifact("outputs/equity_curve.csv")
trades = _load_artifact("outputs/trades.csv")
features = _load_artifact("data/features.csv")
metrics = _load_metrics("outputs/metrics.json")

with st.sidebar:
    if st.button("Run Full Backtest", type="primary"):
        with st.spinner("Running pipeline + backtest..."):
            feature_df = run_features()
            bt = Backtester()
            metrics = bt.run(feature_df)
            equity_curve = _load_artifact("outputs/equity_curve.csv")
            trades = _load_artifact("outputs/trades.csv")
            features = _load_artifact("data/features.csv")
        st.success("Backtest completed.")

with tab1:
    st.subheader("Portfolio Equity Curve")
    if not equity_curve.empty:
        chart_df = equity_curve.copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"])
        st.line_chart(chart_df.set_index("date")["equity"])
    else:
        st.info("Run backtest to generate equity curve.")

    cols = st.columns(4)
    cols[0].metric("Annual Return", f"{100 * metrics.get('annual_return', 0):.2f}%")
    cols[1].metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    cols[2].metric("Max Drawdown", f"{100 * metrics.get('max_drawdown', 0):.2f}%")
    cols[3].metric("Win Rate", f"{100 * metrics.get('win_rate', 0):.2f}%")

with tab2:
    st.subheader("Signal Performance")
    if trades.empty:
        st.info("No trades yet.")
    else:
        by_symbol = trades.groupby(["symbol", "side"]).size().reset_index(name="count")
        st.bar_chart(by_symbol.pivot(index="symbol", columns="side", values="count").fillna(0))
        st.dataframe(trades.tail(30), use_container_width=True)

with tab3:
    st.subheader("Agent Comparison")
    if features.empty:
        st.info("No features available.")
    else:
        sample = features.sort_values("date").groupby("symbol").tail(1)
        orch = Orchestrator()
        rows = []
        for _, r in sample.iterrows():
            out = orch.combine(r)
            for s in out["agent_signals"]:
                rows.append({"symbol": r["symbol"], "agent": s["agent"], "score": s["score"], "signal": s["signal"]})
        comp = pd.DataFrame(rows)
        if not comp.empty:
            st.dataframe(comp, use_container_width=True)

with tab4:
    st.subheader("Live Prediction")
    symbol = st.text_input("Symbol", value="AAPL")
    price = st.number_input("Price", value=180.0)
    momentum_10d = st.number_input("Momentum 10d", value=0.01, step=0.01, format="%.4f")
    momentum_20d = st.number_input("Momentum 20d", value=0.02, step=0.01, format="%.4f")
    sentiment = st.slider("Sentiment Score", min_value=0.0, max_value=1.0, value=0.55)
    vix = st.number_input("VIX", value=18.0, step=0.5)
    if st.button("Generate Prediction"):
        row = pd.Series(
            {
                "symbol": symbol,
                "close": price,
                "momentum_10d": momentum_10d,
                "momentum_20d": momentum_20d,
                "sentiment_score": sentiment,
                "sentiment_momentum": 0.0,
                "zscore_20": 0.0,
                "volume_ratio": 1.0,
                "vix": vix,
                "macro_headwinds": 1.0 if vix > 25 else 0.0,
                "yield_curve_slope": 0.8,
                "price_vs_ma20": 0.01,
            }
        )
        pred = Orchestrator().combine(row)
        st.success(
            f"Signal: {pred['signal']} | Confidence: {pred['confidence']:.2f} | "
            f"Position Size: {100*pred['position_size']:.2f}%"
        )
        st.json(pred)

with tab5:
    st.subheader("Backtest Replay")
    if equity_curve.empty:
        st.info("No replay data available.")
    else:
        replay_df = equity_curve.copy()
        replay_df["date"] = pd.to_datetime(replay_df["date"])
        step = st.slider("Select day", 1, len(replay_df), min(100, len(replay_df)))
        view = replay_df.iloc[:step]
        st.line_chart(view.set_index("date")["equity"])
        latest = view.iloc[-1]
        st.write(f"Date: {latest['date'].date()} | Equity: ${latest['equity']:,.2f}")
