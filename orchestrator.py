"""Signal orchestration and risk management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from agents import BondYieldAgent, MacroRiskAgent, MeanReversionAgent, MomentumAgent, SentimentAgent
from config import SETTINGS


@dataclass
class Orchestrator:
    momentum_agent: MomentumAgent = field(default_factory=MomentumAgent)
    mean_reversion_agent: MeanReversionAgent = field(default_factory=MeanReversionAgent)
    sentiment_agent: SentimentAgent = field(default_factory=SentimentAgent)
    bond_yield_agent: BondYieldAgent = field(default_factory=BondYieldAgent)
    macro_risk_agent: MacroRiskAgent = field(default_factory=MacroRiskAgent)
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "momentum": 0.28,
            "mean_reversion": 0.17,
            "sentiment": 0.22,
            "bond_yield": 0.18,
            "macro_risk": 0.15,
        }
    )

    def combine(self, row: pd.Series, current_drawdown: float = 0.0) -> Dict[str, float | str | List[Dict[str, float | str]]]:
        signals = [
            self.momentum_agent.generate(row),
            self.mean_reversion_agent.generate(row),
            self.sentiment_agent.generate(row),
            self.bond_yield_agent.generate(row),
            self.macro_risk_agent.generate(row),
        ]

        weighted_score = 0.0
        weighted_conf = 0.0
        for s in signals:
            w = self.weights[s["agent"]]
            weighted_score += w * float(s["score"])
            weighted_conf += w * float(s["confidence"])

        final_signal = "HOLD"
        if weighted_score > 0.15:
            final_signal = "BUY"
        elif weighted_score < -0.15:
            final_signal = "SELL"

        macro_signal = next((s for s in signals if s["agent"] == "macro_risk"), {"risk_multiplier": 1.0})
        macro_mult = float(macro_signal.get("risk_multiplier", 1.0))
        drawdown_mult = max(0.2, 1 - current_drawdown / SETTINGS.max_portfolio_drawdown)
        confidence_mult = float(np.clip(weighted_conf, 0.2, 1.0))
        base = SETTINGS.base_position_pct

        position_size = base * macro_mult * drawdown_mult * confidence_mult
        position_size = float(np.clip(position_size, 0.0, SETTINGS.max_position_pct))

        # Hard risk brakes when drawdown breaches limit.
        if current_drawdown >= SETTINGS.max_portfolio_drawdown:
            final_signal = "HOLD"
            position_size = 0.0

        return {
            "signal": final_signal,
            "confidence": float(np.clip(weighted_conf, 0, 1)),
            "score": float(weighted_score),
            "position_size": position_size,
            "agent_signals": signals,
        }
