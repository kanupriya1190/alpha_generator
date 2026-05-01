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
class BondYieldAgent:
    """Yield-aware agent that supports both equities and bond ETFs."""

    bond_symbols: tuple = ("TLT", "IEF", "BND", "AGG")

    def generate(self, row: pd.Series) -> Signal:
        symbol = str(row.get("symbol", "")).upper()
        dgs10 = float(row.get("dgs10", row.get("yield_10y", 3.5)))
        dgs2 = float(row.get("dgs2", row.get("yield_2y", 3.0)))
        fedfunds = float(row.get("fedfunds", 4.5))
        yield_curve = float(row.get("yield_curve_slope", dgs10 - dgs2))
        dgs10_change_21d = float(row.get("yield_10y_change_21d", row.get("dgs10_change_21d", 0.0)))

        is_bond = symbol in self.bond_symbols
        score = 0.0
        if is_bond:
            # Bond ETFs generally benefit when long rates fall.
            score += -1.2 * dgs10_change_21d
            score += -0.25 * max(0.0, dgs10 - fedfunds)
            score += 0.15 if yield_curve < 0 else -0.05
        else:
            # Equities can be pressured by rapid yield rises.
            score += -0.8 * dgs10_change_21d
            score += -0.2 * max(0.0, dgs10 - 4.5)
            score += 0.1 if yield_curve > 0 else -0.1

        signal = _label_from_score(score, buy_thresh=0.12, sell_thresh=-0.12)
        return {"agent": "bond_yield", "signal": signal, "score": score, "confidence": _confidence(score)}


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
