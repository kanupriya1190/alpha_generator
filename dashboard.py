"""Streamlit dashboard for alpha generation project."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from backtester import Backtester
from config import SETTINGS
from features import run_features
from live_trader import AlpacaPaperClient, LiveTrader
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
    if st.button("Recalculate Backtest (Current Portfolio)", type="primary"):
        with st.spinner("Running pipeline + backtest..."):
            feature_df = run_features()
            bt = Backtester()
            metrics = bt.run(feature_df)
            equity_curve = _load_artifact("outputs/equity_curve.csv")
            trades = _load_artifact("outputs/trades.csv")
            features = _load_artifact("data/features.csv")
        st.success("Backtest completed.")
    st.caption("Current symbols: " + ", ".join(SETTINGS.symbols))

with tab1:
    st.subheader("Portfolio Equity Curve")
    metric_symbols = metrics.get("symbols", [])
    if metric_symbols and sorted(metric_symbols) != sorted(SETTINGS.symbols):
        st.warning(
            "Metrics were generated for a different symbol set. "
            "Click 'Recalculate Backtest (Current Portfolio)' to refresh Sharpe/return values."
        )
    if metrics.get("generated_at_utc"):
        st.caption(f"Metrics generated at (UTC): {metrics.get('generated_at_utc')}")

    if not equity_curve.empty:
        chart_df = equity_curve.copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"])
        st.caption("Absolute equity ($)")
        st.line_chart(chart_df.set_index("date")["equity"])

        indexed = chart_df[["date", "equity"]].copy()
        start = float(indexed["equity"].iloc[0]) if len(indexed) else 1.0
        indexed["equity_indexed"] = 100 * indexed["equity"] / max(start, 1e-9)
        st.caption("Indexed equity (starts at 100) to better visualize changes")
        st.line_chart(indexed.set_index("date")["equity_indexed"])
    else:
        st.info("Run backtest to generate equity curve.")

    cols = st.columns(4)
    cols[0].metric("Annual Return", f"{100 * metrics.get('annual_return', 0):.2f}%")
    cols[1].metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    cols[2].metric("Max Drawdown", f"{100 * metrics.get('max_drawdown', 0):.2f}%")
    cols[3].metric("Win Rate", f"{100 * metrics.get('win_rate', 0):.2f}%")
    cols_extra = st.columns(2)
    cols_extra[0].metric("End Equity", f"${metrics.get('end_equity', 0):,.2f}")
    cols_extra[1].metric("Trades", f"{int(metrics.get('num_trades', 0))}")

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
    with st.expander("How BUY / SELL / HOLD is decided"):
        st.markdown(
            """
            - **Momentum Agent**: stronger recent returns -> more bullish score.
            - **Mean Reversion Agent**: extreme z-score (overbought/oversold) -> contrarian score.
            - **Sentiment Agent**: positive sentiment + sentiment momentum -> bullish score.
            - **Macro Risk Agent**: high VIX/macro stress reduces risk and can turn defensive.

            Final decision comes from weighted score:
            - **BUY** when combined score > `0.15`
            - **SELL** when combined score < `-0.15`
            - **HOLD** otherwise

            In live paper trading, we then convert this into **target portfolio weights**, optionally
            apply **stats-arbitrage pair adjustments**, and send rebalance orders.
            """
        )

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

    st.divider()
    st.subheader("Alpaca Paper Trading")
    client = AlpacaPaperClient()
    if not client.enabled:
        st.warning("Missing `ALPACA_API_KEY` or `ALPACA_SECRET_KEY` in `.env`.")
    else:
        st.caption("Run controls")
        use_stats_arb = st.checkbox("Enable stats-arbitrage allocation tweaks", value=True)
        rebalance = st.checkbox("Enable target-weight rebalancing", value=True)
        action_col1, action_col2 = st.columns(2)
        run_dry = action_col1.button("Run Signal Scan (Dry Run)")
        run_live = action_col2.button("Execute Paper Orders")

        result = None
        if run_dry:
            with st.spinner("Running dry-run signal scan..."):
                result = LiveTrader().run_once(dry_run=True, use_stats_arb=use_stats_arb, rebalance=rebalance)
        if run_live:
            with st.spinner("Submitting market orders to Alpaca paper account..."):
                result = LiveTrader().run_once(dry_run=False, use_stats_arb=use_stats_arb, rebalance=rebalance)

        try:
            account = client.get_account()
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Equity", f"${float(account.get('equity', 0)):,.2f}")
            a2.metric("Buying Power", f"${float(account.get('buying_power', 0)):,.2f}")
            a3.metric("Cash", f"${float(account.get('cash', 0)):,.2f}")
            a4.metric("Status", str(account.get("status", "unknown")).upper())
        except Exception as exc:
            st.error(f"Could not load Alpaca account: {exc}")

        try:
            positions = pd.DataFrame(client.list_positions())
            st.markdown("**Open Positions**")
            if positions.empty:
                st.info("No open positions.")
            else:
                keep_cols = [c for c in ["symbol", "qty", "avg_entry_price", "market_value", "unrealized_pl"] if c in positions.columns]
                st.dataframe(positions[keep_cols], use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load positions: {exc}")

        try:
            orders = pd.DataFrame(client.list_orders(status="all", limit=25))
            st.markdown("**Recent Orders**")
            if orders.empty:
                st.info("No recent orders.")
            else:
                keep_cols = [
                    c
                    for c in ["submitted_at", "symbol", "side", "qty", "type", "status", "filled_avg_price"]
                    if c in orders.columns
                ]
                st.dataframe(orders[keep_cols], use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load orders: {exc}")

        if result is not None:
            st.markdown("**Latest Live-Trader Actions**")
            st.dataframe(pd.DataFrame(result["actions"]), use_container_width=True)
            if result.get("target_weights"):
                st.markdown("**Target Weights (%)**")
                tw = pd.DataFrame(
                    [{"symbol": k, "target_weight_pct": v} for k, v in result["target_weights"].items()]
                ).sort_values("target_weight_pct", ascending=False)
                st.dataframe(tw, use_container_width=True)
            if result.get("stats_arb"):
                st.markdown("**Stats-Arbitrage Diagnostics**")
                st.dataframe(pd.DataFrame(result["stats_arb"]), use_container_width=True)

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
