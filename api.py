"""FastAPI inference endpoint for live signal prediction."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from orchestrator import Orchestrator

app = FastAPI(title="Alpha Generator API", version="1.0.0")
orchestrator = Orchestrator()


class PredictRequest(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    price: float
    momentum_10d: float = 0.0
    momentum_20d: float = 0.0
    zscore_20: float = 0.0
    volume_ratio: float = 1.0
    sentiment_score: float = 0.5
    sentiment_momentum: float = 0.0
    vix: float = 20.0
    macro_headwinds: float = 0.0
    yield_curve_slope: float = 1.0
    price_vs_ma20: float = 0.0


class PredictResponse(BaseModel):
    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    position_size: str
    score: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    row = pd.Series(
        {
            "symbol": payload.symbol,
            "close": payload.price,
            "momentum_10d": payload.momentum_10d,
            "momentum_20d": payload.momentum_20d,
            "zscore_20": payload.zscore_20,
            "volume_ratio": payload.volume_ratio,
            "sentiment_score": payload.sentiment_score,
            "sentiment_momentum": payload.sentiment_momentum,
            "vix": payload.vix,
            "macro_headwinds": payload.macro_headwinds,
            "yield_curve_slope": payload.yield_curve_slope,
            "price_vs_ma20": payload.price_vs_ma20,
        }
    )
    out = orchestrator.combine(row, current_drawdown=0.0)
    return PredictResponse(
        signal=out["signal"],  # type: ignore[arg-type]
        confidence=float(out["confidence"]),
        position_size=f"{100 * float(out['position_size']):.2f}%",
        score=float(out["score"]),
    )
