"""Rule-based signal agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import pandas as pd

Signal = Dict[str, Union[float, str]]


def _label_from_score(score: float, buy_thresh: float = 0.2, sell_thresh: float = -0.2) -> str:
    if score >= buy_thresh:
        return "BUY"
    if score <= sell_thresh:
        return "SELL"
    return "HOLD"


def _confidence(score: float) -> float:
    return float(np.clip(abs(score), 0, 1))


@dataclass
class MomentumAgent:
    """Trend-following signal generator."""

    def generate(self, row: pd.Series) -> Signal:
        score = 0.0
        score += 1.5 * float(row.get("momentum_10d", 0))
        score += 1.0 * float(row.get("momentum_20d", 0))
        score += 0.5 * float(row.get("price_vs_ma20", 0))
        signal = _label_from_score(score)
        return {"agent": "momentum", "signal": signal, "score": score, "confidence": _confidence(score)}


@dataclass
class MeanReversionAgent:
    """Contrarian signal generator."""

    def generate(self, row: pd.Series) -> Signal:
        zscore = float(row.get("zscore_20", 0))
        vol_ratio = float(row.get("volume_ratio", 1))
        score = 0.0
        if zscore > 2 and vol_ratio < 1:
            score = -min(1.0, (zscore - 2) / 2 + 0.2)
        elif zscore < -2 and vol_ratio < 1:
            score = min(1.0, (-zscore - 2) / 2 + 0.2)
        signal = _label_from_score(score)
        return {"agent": "mean_reversion", "signal": signal, "score": score, "confidence": _confidence(score)}


@dataclass
class SentimentAgent:
    """Sentiment + momentum blended signal generator."""

    def generate(self, row: pd.Series) -> Signal:
        sentiment = float(row.get("sentiment_score", 0.5))
        sent_momo = float(row.get("sentiment_momentum", 0))
        momentum = float(row.get("momentum_10d", 0))
        score = (sentiment - 0.5) * 1.8 + 0.8 * sent_momo + 0.7 * momentum
        signal = _label_from_score(score)
        return {"agent": "sentiment", "signal": signal, "score": score, "confidence": _confidence(score)}


@dataclass
class MacroRiskAgent:
    """Market regime and risk scaler."""

    def generate(self, row: pd.Series) -> Signal:
        vix = float(row.get("vix", 20))
        macro_headwinds = float(row.get("macro_headwinds", 0))
        yield_curve = float(row.get("yield_curve_slope", 0))
        score = 0.0
        if vix > 25:
            score -= 0.7
        if macro_headwinds > 0:
            score -= 0.4
        if yield_curve > 0:
            score += 0.2
        signal = _label_from_score(score, buy_thresh=0.3, sell_thresh=-0.3)
        risk_multiplier = 1.0
        if vix > 25:
            risk_multiplier *= 0.6
        if vix > 35:
            risk_multiplier *= 0.5
        if macro_headwinds > 0:
            risk_multiplier *= 0.8
        return {
            "agent": "macro_risk",
            "signal": signal,
            "score": score,
            "confidence": _confidence(score),
            "risk_multiplier": float(np.clip(risk_multiplier, 0.1, 1.0)),
        }
