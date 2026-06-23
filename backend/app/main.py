import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    BacktestRequest,
    BacktestResponse,
    ComponentVarBreakdown,
    MonteCarloPathsRequest,
    MonteCarloPathsResponse,
    PortfolioTickerRequest,
    PortfolioTickerResponse,
    PortfolioVarRequest,
    PortfolioVarResponse,
    TickerRequest,
    TickerResponse,
    VarRequest,
    VarResponse,
)
from app.var_engine import (
    backtest,
    compute_component_var,
    compute_correlation_matrix,
    compute_incremental_var,
    compute_marginal_var,
    compute_portfolio_returns,
    compute_portfolio_var_surface,
    compute_return_distribution,
    compute_return_stats,
    compute_var_surface,
    portfolio_var_parametric,
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
# Portfolio: fetch multiple tickers
# ---------------------------------------------------------------------------

@app.post("/api/portfolio-tickers", response_model=PortfolioTickerResponse)
async def fetch_portfolio_tickers(req: PortfolioTickerRequest) -> PortfolioTickerResponse:
    import yfinance as yf  # lazy - heavy import

    all_histories = {}
    for asset in req.assets:
        tk = yf.Ticker(asset.ticker)
        hist = tk.history(period=req.period)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for '{asset.ticker}'")
        all_histories[asset.ticker.upper()] = hist

    # align on shared trading dates (inner join)
    common_dates = None
    for hist in all_histories.values():
        dates = set(hist.index)
        common_dates = dates if common_dates is None else common_dates & dates
    common_dates = sorted(common_dates)

    if len(common_dates) < 30:
        raise HTTPException(status_code=422, detail="Fewer than 30 overlapping trading days")

    tickers = []
    asset_returns = []
    weights = []
    for asset in req.assets:
        t = asset.ticker.upper()
        hist = all_histories[t]
        prices = hist.loc[common_dates, "Close"].values
        log_ret = np.log(prices[1:] / prices[:-1])
        tickers.append(t)
        asset_returns.append(log_ret.tolist())
        weights.append(asset.weight)

    dates = [d.strftime("%Y-%m-%d") for d in common_dates[1:]]

    return PortfolioTickerResponse(
        tickers=tickers,
        asset_returns=asset_returns,
        dates=dates,
        weights=weights,
    )


# ---------------------------------------------------------------------------
# Portfolio: VaR computation
# ---------------------------------------------------------------------------

@app.post("/api/portfolio-var", response_model=PortfolioVarResponse)
async def compute_portfolio_var(req: PortfolioVarRequest) -> PortfolioVarResponse:
    if len(req.asset_returns) != len(req.weights):
        raise HTTPException(status_code=422, detail="asset_returns and weights must have the same length")
    if len(req.asset_returns) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 assets")

    assets = [np.array(r, dtype=float) for r in req.asset_returns]
    w = np.array(req.weights, dtype=float)

    # normalize weights to sum to 1
    w = w / w.sum()

    port_returns = compute_portfolio_returns(assets, w)

    stats = compute_return_stats(port_returns)
    surface = compute_portfolio_var_surface(assets, w, req.confidences, req.holding_periods, req.n_simulations)
    distribution = compute_return_distribution(port_returns)
    corr = compute_correlation_matrix(assets)

    # risk decomposition at 95% confidence, 1-day
    comp_var = compute_component_var(assets, w, 0.95)
    marg_var = compute_marginal_var(assets, w, 0.95)
    inc_var = compute_incremental_var(assets, w, 0.95)
    port_var_95, _ = portfolio_var_parametric(assets, w, 0.95, 1)

    # standalone VaR for each asset (weighted by its portfolio share)
    standalone_vars = []
    for i, r in enumerate(assets):
        sv, _ = portfolio_var_parametric([r], np.array([1.0]), 0.95, 1)
        standalone_vars.append(w[i] * sv)

    breakdown = []
    for i, t in enumerate(req.tickers):
        breakdown.append(ComponentVarBreakdown(
            ticker=t,
            weight=float(w[i]),
            standalone_var=standalone_vars[i],
            component_var=comp_var[i],
            marginal_var=marg_var[i],
            incremental_var=inc_var[i],
            pct_contribution=comp_var[i] / port_var_95 if port_var_95 != 0 else 0.0,
        ))

    diversification_benefit = sum(standalone_vars) - port_var_95

    return PortfolioVarResponse(
        stats=stats,
        var_surface=surface,
        distribution=distribution,
        correlation_matrix=corr.tolist(),
        tickers=req.tickers,
        weights=w.tolist(),
        component_breakdown=breakdown,
        diversification_benefit=diversification_benefit,
    )


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
