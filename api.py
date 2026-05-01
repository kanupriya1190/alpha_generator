"""FastAPI inference endpoint for live signal prediction."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from live_trader import LiveTrader
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
    dgs10: float = 4.0
    dgs2: float = 3.8
    fedfunds: float = 4.5
    yield_10y_change_21d: float = 0.0
    price_vs_ma20: float = 0.0


class PredictResponse(BaseModel):
    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    position_size: str
    score: float


class PaperRunRequest(BaseModel):
    dry_run: bool = True
    use_stats_arb: bool = True
    rebalance: bool = True


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
            "dgs10": payload.dgs10,
            "dgs2": payload.dgs2,
            "fedfunds": payload.fedfunds,
            "yield_10y_change_21d": payload.yield_10y_change_21d,
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


@app.get("/paper/account")
def paper_account() -> dict:
    try:
        trader = LiveTrader()
        return trader.client.get_account()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/paper/orders")
def paper_orders(limit: int = 25) -> list[dict]:
    try:
        trader = LiveTrader()
        return trader.client.list_orders(status="all", limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/paper/positions")
def paper_positions() -> list[dict]:
    try:
        trader = LiveTrader()
        return trader.client.list_positions()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/paper/run-once")
def paper_run_once(payload: PaperRunRequest) -> dict:
    try:
        trader = LiveTrader()
        return trader.run_once(
            dry_run=payload.dry_run,
            use_stats_arb=payload.use_stats_arb,
            rebalance=payload.rebalance,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
