"""Event-style backtester for the multi-agent system."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from config import SETTINGS
from features import run_features
from orchestrator import Orchestrator


@dataclass
class Position:
    quantity: float = 0.0
    entry_price: float = 0.0


@dataclass
class Backtester:
    orchestrator: Orchestrator = field(default_factory=Orchestrator)

    def run(self, features: pd.DataFrame) -> Dict[str, Any]:
        SETTINGS.ensure_dirs()
        df = features.sort_values(["date", "symbol"]).copy()
        dates = sorted(df["date"].unique())

        cash = SETTINGS.initial_capital
        positions: Dict[str, Position] = {sym: Position() for sym in SETTINGS.symbols}
        equity_records: List[Dict[str, float | str]] = []
        trade_records: List[Dict[str, float | str]] = []

        equity_peak = SETTINGS.initial_capital
        prev_equity = SETTINGS.initial_capital
        daily_returns = []

        for dt in dates:
            day_rows = df[df["date"] == dt]

            # Mark-to-market before new trades.
            mtm_value = 0.0
            close_by_symbol = {r["symbol"]: float(r["close"]) for _, r in day_rows.iterrows()}
            for sym, pos in positions.items():
                px = close_by_symbol.get(sym)
                if px is not None:
                    mtm_value += pos.quantity * px
            equity_before = cash + mtm_value
            equity_peak = max(equity_peak, equity_before)
            current_drawdown = (equity_peak - equity_before) / max(equity_peak, 1e-9)

            for _, row in day_rows.iterrows():
                sym = row["symbol"]
                close = float(row["close"])
                decision = self.orchestrator.combine(row, current_drawdown=current_drawdown)
                signal = str(decision["signal"])
                target_weight = float(decision["position_size"])

                pos = positions[sym]
                current_value = pos.quantity * close
                target_value = equity_before * target_weight if signal == "BUY" else 0.0 if signal == "SELL" else current_value
                delta_value = target_value - current_value
                if abs(delta_value) < 1.0:
                    continue

                side = "BUY" if delta_value > 0 else "SELL"
                exec_price = close * (1 + SETTINGS.slippage_rate if side == "BUY" else 1 - SETTINGS.slippage_rate)
                qty = abs(delta_value) / max(exec_price, 1e-9)
                notional = qty * exec_price
                fee = notional * SETTINGS.fee_rate

                # Stop-loss: flatten if down >= 2% from entry.
                if pos.quantity > 0 and close <= pos.entry_price * (1 - SETTINGS.stop_loss_pct):
                    side = "SELL"
                    qty = pos.quantity
                    notional = qty * exec_price
                    fee = notional * SETTINGS.fee_rate

                if side == "BUY":
                    total_cost = notional + fee
                    if total_cost <= cash:
                        cash -= total_cost
                        new_qty = pos.quantity + qty
                        pos.entry_price = (pos.entry_price * pos.quantity + exec_price * qty) / max(new_qty, 1e-9)
                        pos.quantity = new_qty
                    else:
                        continue
                else:
                    qty = min(qty, pos.quantity)
                    proceeds = qty * exec_price - fee
                    cash += proceeds
                    pos.quantity -= qty
                    if pos.quantity <= 1e-9:
                        pos.quantity = 0.0
                        pos.entry_price = 0.0

                trade_records.append(
                    {
                        "date": dt,
                        "symbol": sym,
                        "side": side,
                        "qty": qty,
                        "exec_price": exec_price,
                        "notional": notional,
                        "fee": fee,
                        "signal": signal,
                        "confidence": decision["confidence"],
                    }
                )

            # End-of-day equity.
            eod_positions = 0.0
            for sym, pos in positions.items():
                px = close_by_symbol.get(sym)
                if px is not None:
                    eod_positions += pos.quantity * px
            equity = cash + eod_positions
            daily_ret = (equity / max(prev_equity, 1e-9)) - 1
            daily_returns.append(daily_ret)
            prev_equity = equity
            equity_peak = max(equity_peak, equity)
            drawdown = (equity_peak - equity) / max(equity_peak, 1e-9)
            equity_records.append({"date": dt, "equity": equity, "drawdown": drawdown})

        equity_curve = pd.DataFrame(equity_records)
        trades = pd.DataFrame(trade_records)
        metrics = self._metrics(equity_curve, daily_returns, trades)
        metrics.update(
            {
                "symbols": SETTINGS.symbols,
                "start_date": str(SETTINGS.start_date),
                "end_date": str(SETTINGS.end_date),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

        equity_curve.to_csv(SETTINGS.equity_curve_path, index=False)
        trades.to_csv(SETTINGS.trades_path, index=False)
        with open(SETTINGS.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    @staticmethod
    def _metrics(equity_curve: pd.DataFrame, daily_returns: List[float], trades: pd.DataFrame) -> Dict[str, float]:
        if equity_curve.empty:
            return {}
        start_val = float(equity_curve["equity"].iloc[0])
        end_val = float(equity_curve["equity"].iloc[-1])
        years = max(len(equity_curve) / 252.0, 1e-9)
        annual_return = (end_val / max(start_val, 1e-9)) ** (1 / years) - 1

        rets = np.array(daily_returns, dtype=float)
        sharpe = 0.0
        if rets.std() > 1e-12:
            sharpe = np.sqrt(252) * rets.mean() / rets.std()

        max_drawdown = float(equity_curve["drawdown"].max()) if "drawdown" in equity_curve else 0.0

        pnl = []
        if not trades.empty:
            signed_notional = np.where(trades["side"] == "SELL", trades["notional"], -trades["notional"])
            pnl = signed_notional - trades["fee"]

        win_rate = 0.0
        profit_factor = 0.0
        if len(pnl) > 0:
            pnl = np.array(pnl, dtype=float)
            wins = pnl[pnl > 0].sum()
            losses = -pnl[pnl < 0].sum()
            win_rate = float((pnl > 0).mean())
            profit_factor = float(wins / max(losses, 1e-9))

        return {
            "start_equity": start_val,
            "end_equity": end_val,
            "annual_return": float(annual_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": max_drawdown,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "num_trades": int(len(trades)),
        }


def main() -> None:
    features = run_features()
    backtester = Backtester()
    metrics = backtester.run(features)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
