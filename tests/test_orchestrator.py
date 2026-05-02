import pandas as pd

from orchestrator import Orchestrator


def test_orchestrator_outputs_valid_decision_and_size() -> None:
    orch = Orchestrator()
    row = pd.Series(
        {
            "symbol": "NVDA",
            "momentum_10d": 0.12,
            "momentum_20d": 0.20,
            "price_vs_ma20": 0.08,
            "zscore_20": -0.5,
            "volume_ratio": 1.1,
            "sentiment_score": 0.82,
            "sentiment_momentum": 0.10,
            "dgs10": 3.8,
            "dgs2": 3.5,
            "fedfunds": 4.5,
            "yield_curve_slope": 0.3,
            "yield_10y_change_21d": -0.04,
            "vix": 16.0,
            "macro_headwinds": 0.0,
        }
    )

    out = orch.combine(row)
    assert out["signal"] in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= float(out["confidence"]) <= 1.0
    assert 0.0 <= float(out["position_size"]) <= 0.25
    assert len(out["agent_signals"]) == 5


def test_orchestrator_drawdown_guard_zeros_position() -> None:
    orch = Orchestrator()
    row = pd.Series(
        {
            "symbol": "NVDA",
            "momentum_10d": 0.2,
            "momentum_20d": 0.2,
            "price_vs_ma20": 0.1,
            "sentiment_score": 0.8,
            "sentiment_momentum": 0.1,
            "zscore_20": 0.0,
            "volume_ratio": 1.0,
            "vix": 20.0,
            "macro_headwinds": 0.0,
            "yield_curve_slope": 0.2,
        }
    )

    out = orch.combine(row, current_drawdown=0.20)
    assert out["signal"] == "HOLD"
    assert float(out["position_size"]) == 0.0
