# Multi-Agent Alpha Generator

End-to-end trading system that ingests market/macro/sentiment data, engineers features, generates signals with 4 specialized agents, runs a realistic backtest, serves live predictions, and visualizes performance in Streamlit.

## Architecture

- `data_pipeline.py` - Market/macro/sentiment ingestion and data validation
- `features.py` - 30+ engineered features (momentum, volatility, macro, sentiment)
- `agents.py` - Momentum, Mean Reversion, Sentiment, Macro Risk agents
- `orchestrator.py` - Weighted signal ensemble + position sizing + drawdown brakes
- `backtester.py` - Event-style 2020-2025 walk-forward simulation with fees/slippage
- `api.py` - FastAPI service (`POST /predict`)
- `dashboard.py` - Streamlit dashboard (results, signal analysis, replay, live prediction)
- `backtest.py` - One-command local backtest runner

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set your keys in `.env`:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- optional: `FRED_API_KEY`, `NEWS_API_KEY`

Run full pipeline + backtest:

```bash
python backtest.py
```

Run API:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Run dashboard:

```bash
streamlit run dashboard.py
```

## API Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol":"AAPL",
    "price":185.0,
    "momentum_10d":0.03,
    "momentum_20d":0.02,
    "sentiment_score":0.72,
    "vix":19.5
  }'
```

## Outputs

- `data/features.csv` - full feature set
- `outputs/equity_curve.csv` - day-level portfolio value and drawdown
- `outputs/trades.csv` - trade log
- `outputs/metrics.json` - annual return, Sharpe, max drawdown, win rate

## Notes

- The system prefers Alpaca bars when credentials are present and falls back to yfinance.
- Backtest includes slippage (0.10%) and fees (0.05%).
- Secrets are loaded from environment variables; never commit `.env`.
