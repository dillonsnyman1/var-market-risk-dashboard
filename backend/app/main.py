import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    BacktestRequest,
    BacktestResponse,
    MonteCarloPathsRequest,
    MonteCarloPathsResponse,
    TickerRequest,
    TickerResponse,
    VarRequest,
    VarResponse,
)
from app.var_engine import (
    backtest,
    compute_return_distribution,
    compute_return_stats,
    compute_var_surface,
    simulate_paths,
)

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")

app = FastAPI(title="VaR & Market Risk Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Ticker data
# ---------------------------------------------------------------------------

@app.post("/api/ticker", response_model=TickerResponse)
async def fetch_ticker(req: TickerRequest) -> TickerResponse:
    import yfinance as yf  # lazy - heavy import, only needed for this endpoint

    tk = yf.Ticker(req.ticker)
    hist = tk.history(period=req.period)

    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{req.ticker}'")

    prices = hist["Close"].values
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]
    log_ret = np.log(prices[1:] / prices[:-1])

    return TickerResponse(
        ticker=req.ticker.upper(),
        returns=log_ret.tolist(),
        dates=dates[1:],
        prices=prices.tolist(),
    )


# ---------------------------------------------------------------------------
# CSV upload
# ---------------------------------------------------------------------------

@app.post("/api/upload", response_model=TickerResponse)
async def upload_csv(file: UploadFile) -> TickerResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="File must be a CSV")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {exc}")

    # case-insensitive column lookup
    col_lower = {c.lower(): c for c in df.columns}

    # prefer a returns column if present, otherwise compute from prices
    for name in ["return", "log_return", "returns", "log_returns"]:
        if name in col_lower:
            returns = df[col_lower[name]].dropna().values
            return TickerResponse(
                ticker="CSV",
                returns=returns.tolist(),
                dates=[str(i) for i in range(len(returns))],
                prices=[],
            )

    # no returns column found - try prices instead
    for name in ["close", "adj_close", "adj close", "price", "prices"]:
        if name in col_lower:
            prices = df[col_lower[name]].dropna().values
            if len(prices) < 2:
                raise HTTPException(status_code=422, detail="Need at least 2 prices")
            log_ret = np.log(prices[1:] / prices[:-1])
            return TickerResponse(
                ticker="CSV",
                returns=log_ret.tolist(),
                dates=[str(i) for i in range(len(log_ret))],
                prices=prices.tolist(),
            )

    raise HTTPException(
        status_code=422,
        detail="CSV must have a column named 'return', 'close', 'price', or similar",
    )


# ---------------------------------------------------------------------------
# VaR computation
# ---------------------------------------------------------------------------

@app.post("/api/var", response_model=VarResponse)
async def compute_var(req: VarRequest) -> VarResponse:
    r = np.array(req.returns, dtype=float)
    if len(r) < 30:
        raise HTTPException(status_code=422, detail="Need at least 30 returns")

    stats = compute_return_stats(r)
    surface = compute_var_surface(r, req.confidences, req.holding_periods, req.n_simulations)
    distribution = compute_return_distribution(r)

    return VarResponse(
        stats=stats,
        var_surface=surface,
        distribution=distribution,
    )


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

@app.post("/api/backtest", response_model=BacktestResponse)
async def run_backtest(req: BacktestRequest) -> BacktestResponse:
    r = np.array(req.returns, dtype=float)
    if len(r) < req.window + 10:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {req.window + 10} returns for a {req.window}-day window",
        )

    result = backtest(r, req.confidence, req.window, req.method.value)
    return BacktestResponse(**result)


# ---------------------------------------------------------------------------
# Monte Carlo paths
# ---------------------------------------------------------------------------

@app.post("/api/monte-carlo-paths", response_model=MonteCarloPathsResponse)
async def monte_carlo_paths(req: MonteCarloPathsRequest) -> MonteCarloPathsResponse:
    r = np.array(req.returns, dtype=float)
    if len(r) < 30:
        raise HTTPException(status_code=422, detail="Need at least 30 returns")

    result = simulate_paths(r, req.holding_period, req.n_simulations)
    return MonteCarloPathsResponse(**result)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

from mangum import Mangum

handler = Mangum(app)
