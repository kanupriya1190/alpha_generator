"""Generate a public, GitHub-friendly snapshot for docs/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from config import SETTINGS
from live_trader import AlpacaPaperClient


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_live_run(output_dir: Path) -> Dict[str, Any]:
    run_files = sorted(output_dir.glob("run_*.json"))
    if not run_files:
        return {}
    latest = run_files[-1]
    data = _load_json(latest)
    actions = data.get("actions", [])
    simplified = []
    for a in actions[:20]:
        simplified.append(
            {
                "symbol": a.get("symbol"),
                "signal": a.get("signal"),
                "order_side": a.get("order_side"),
                "order_qty": a.get("order_qty"),
                "target_weight_pct": round(_safe_float(a.get("target_weight_pct")), 4),
                "sentiment_source": a.get("sentiment_source"),
                "news_headline_count": int(_safe_float(a.get("news_headline_count"), 0)),
            }
        )
    return {
        "file": latest.name,
        "actions": simplified,
    }


def _account_snapshot(client: AlpacaPaperClient) -> Dict[str, Any]:
    if not client.enabled:
        return {"status": "disabled", "error": "Missing Alpaca credentials"}
    try:
        account = client.get_account()
        return {
            "status": str(account.get("status", "unknown")).upper(),
            "equity": _safe_float(account.get("equity")),
            "cash": _safe_float(account.get("cash")),
            "buying_power": _safe_float(account.get("buying_power")),
            "daytrade_count": int(_safe_float(account.get("daytrade_count"), 0)),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _positions_snapshot(client: AlpacaPaperClient) -> List[Dict[str, Any]]:
    try:
        positions = client.list_positions()
    except Exception as exc:
        return [{"error": str(exc)}]
    out = []
    for p in positions:
        out.append(
            {
                "symbol": p.get("symbol"),
                "qty": _safe_float(p.get("qty")),
                "avg_entry_price": _safe_float(p.get("avg_entry_price")),
                "market_value": _safe_float(p.get("market_value")),
                "unrealized_pl": _safe_float(p.get("unrealized_pl")),
            }
        )
    return out


def _orders_snapshot(client: AlpacaPaperClient, limit: int = 10) -> List[Dict[str, Any]]:
    try:
        orders = client.list_orders(status="all", limit=limit)
    except Exception as exc:
        return [{"error": str(exc)}]
    out = []
    for o in orders[:limit]:
        out.append(
            {
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "qty": _safe_float(o.get("qty")),
                "status": o.get("status"),
                "filled_avg_price": _safe_float(o.get("filled_avg_price")),
                "submitted_at": o.get("submitted_at"),
            }
        )
    return out


def generate_snapshot() -> Dict[str, Any]:
    metrics = _load_json(SETTINGS.metrics_path)
    client = AlpacaPaperClient()

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "Multi-Agent Alpha Generator",
        "symbols": SETTINGS.symbols,
        "backtest_metrics": {
            "annual_return": metrics.get("annual_return"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "max_drawdown": metrics.get("max_drawdown"),
            "win_rate": metrics.get("win_rate"),
            "profit_factor": metrics.get("profit_factor"),
            "num_trades": metrics.get("num_trades"),
            "start_date": metrics.get("start_date"),
            "end_date": metrics.get("end_date"),
            "generated_at_utc": metrics.get("generated_at_utc"),
        },
        "paper_account": _account_snapshot(client),
        "positions": _positions_snapshot(client),
        "recent_orders": _orders_snapshot(client, limit=10),
        "latest_live_run": _latest_live_run(Path("outputs/live_runs")),
    }
    return payload


def main() -> None:
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    payload = generate_snapshot()
    out_path = docs_dir / "live_snapshot.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved snapshot to {out_path}")


if __name__ == "__main__":
    main()
