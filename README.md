# VaR & Market Risk Dashboard

[![CI/CD](https://github.com/dillonsnyman1/var-market-risk-dashboard/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/dillonsnyman1/var-market-risk-dashboard/actions/workflows/ci-cd.yml)

A full-stack demo that calculates single-asset Value at Risk (VaR) and
Expected Shortfall (CVaR) using three methods side-by-side, with
backtesting and Monte Carlo path simulation.

- **Backend**: Python + FastAPI for the VaR calculation engine
- **Frontend**: React + Vite + TypeScript dashboard (VaR surface, return
  distribution, Monte Carlo paths, backtesting)

> **Live demo**: *Coming soon - will be deployed to CloudFront once infrastructure is in place.*
>
> The backend is fully stateless: all VaR computation is done in-request
> with no database or storage of any kind.

> **Disclaimer**: Simplified demo built for portfolio purposes. Not a
> production risk system and should not be used for regulatory reporting or
> live risk management. All defaults are synthetic.

---

## Background

Value at Risk (VaR) answers a simple question: "What is the most I can
expect to lose over a given time horizon, at a given confidence level, under
normal market conditions?" A 95% one-day VaR of 2.3% means that on 95% of
trading days, the portfolio loss will not exceed 2.3%.

VaR is the standard market risk metric used by banks, asset managers and
regulators. Basel II/III require banks to hold capital against market risk
based on VaR (and more recently, Expected Shortfall). Despite its
limitations - it says nothing about the size of losses beyond the threshold,
and its accuracy depends entirely on the assumptions you feed it - VaR
remains the starting point for any market risk conversation.

Expected Shortfall (ES), also called Conditional VaR (CVaR), addresses the
main limitation: it measures the average loss in the tail beyond the VaR
threshold. ES is always greater than or equal to VaR and is a coherent risk
measure (VaR is not, because it can violate sub-additivity for non-normal
distributions). The Basel Committee's Fundamental Review of the Trading Book
(FRTB) replaced VaR with ES as the primary capital metric, though VaR
remains central to internal risk management.

Three approaches are compared here:

- **Historical Simulation** uses the actual observed return distribution
  directly. VaR is simply the empirical percentile of past returns. No
  distributional assumptions, but it implicitly assumes the past is
  representative of the future and is slow to adapt to regime changes.

- **Variance-Covariance (Parametric)** assumes returns are normally
  distributed and computes VaR analytically from the estimated mean and
  standard deviation. Fast and transparent, but the normal assumption
  underestimates tail risk - real return distributions exhibit fat tails
  (excess kurtosis) and negative skew.

- **Monte Carlo Simulation** estimates return parameters from historical
  data and simulates thousands of possible future paths using Geometric
  Brownian Motion (GBM). More flexible than the parametric approach (you
  can swap in any dynamics - stochastic vol, jumps, mean reversion) at the
  cost of computational time and sampling noise.

The backtesting module validates each VaR model against realised returns:
at each point in time, it computes VaR using a trailing window and checks
whether the next-day loss exceeded the prediction. If the breach rate is
significantly higher than expected, the model is miscalibrated - the Kupiec
proportion-of-failures (POF) test formalises this as a likelihood ratio
statistic.

---

## Methodology

### 1. Historical Simulation

The simplest approach: sort the observed returns and read off the percentile.

```
VaR(α) = -Percentile(returns, 1 - α)
CVaR(α) = -Mean(returns where return ≤ -VaR)
```

For multi-day VaR, the one-day figure is scaled by √h (the square-root-of-
time rule), which assumes i.i.d. returns. This is a simplification - returns
exhibit serial correlation, volatility clustering, and other dependencies -
but it is the standard industry approximation and avoids the need to
estimate multi-day return distributions directly.

### 2. Variance-Covariance (Parametric)

Assumes returns ~ N(μ, σ²) and computes VaR analytically.

```
VaR(α) = -(μ·h - z_α · σ·√h)
CVaR(α) = -(μ·h - σ·√h · φ(z_α) / (1 - α))
```

where z_α is the standard normal quantile (e.g. 1.6449 for 95%), φ is the
standard normal PDF, μ is scaled by the holding period h, and σ is scaled
by √h.

The main limitation is the normality assumption. Equity returns typically
exhibit excess kurtosis (fat tails) and negative skew, which means the
parametric VaR underestimates tail risk. The return distribution chart
overlays the fitted normal against the empirical histogram so the deviation
is visible.

### 3. Monte Carlo Simulation

Estimates μ and σ from historical returns and simulates future price paths
via Geometric Brownian Motion:

```
S_t = S_0 · exp((μ - σ²/2)·t + σ·√t·Z),   Z ~ N(0,1)
```

For each of n simulations, the cumulative return over the holding period is
computed. VaR and CVaR are then the percentile and conditional mean of the
simulated return distribution, identical to the historical method but
applied to simulated rather than observed data.

The Monte Carlo tab shows a sample of simulated paths and a histogram of
terminal returns with VaR thresholds marked.

### 4. Backtesting

Rolling window backtest: at each date t, VaR is estimated from the trailing
w observations and compared against the actual return at t+1.

```
Breach: actual_loss > VaR_predicted
Breach rate: n_breaches / n_observations
Expected rate: 1 - confidence
```

The Kupiec POF test checks whether the observed breach rate is consistent
with the expected rate:

```
LR = -2 · ln[(1-p)^(n-x) · p^x / (1-p̂)^(n-x) · p̂^x]
```

where p = 1 - confidence (expected breach probability), p̂ = x/n (observed),
x = breach count, n = total observations. Under H₀ (model is correctly
calibrated), LR ~ χ²(1). A low p-value indicates the model is producing too
many or too few breaches.

---

## Roadmap

This project is being built incrementally in three phases:

### Phase 1: Reference implementations *(in progress)*

Standalone, side-by-side implementations of the core VaR algorithms in
Python, C++, R, MATLAB and SAS - each idiomatic to its language, all
validated against the same shared fixture data. No web framework or
frontend. See [`reference/`](reference/).

### Phase 2: Full-stack local demo

FastAPI backend exposing the VaR engine as an API, with a React + Vite +
TypeScript dashboard for interactive exploration. Supports both live
ticker data (via Yahoo Finance) and user-uploaded CSV files. Runs locally
with no cloud dependencies.

### Phase 3: AWS deployment

Terraform infrastructure (Lambda, API Gateway, S3, CloudFront) and a
GitHub Actions CI/CD pipeline - same architecture as the other projects
in this portfolio. Deployed automatically on every push to `main`.

### Future: Multi-asset portfolio expansion

Portfolio VaR with correlation modelling, component/marginal VaR
decomposition, diversification benefit analysis, stress testing, and
conditional volatility (GARCH). Tracked in [EXTENSIONS.md](EXTENSIONS.md).

---

## Known limitations and possible extensions

> Tracked in [EXTENSIONS.md](EXTENSIONS.md).

### Model limitations

- **Single asset only.** The current implementation computes VaR for a
  single return series. Portfolio VaR requires modelling the correlation
  structure across multiple assets - see the extensions doc for the planned
  multi-asset expansion.

- **Constant volatility.** All three methods estimate a single σ from
  historical data. In reality, volatility clusters (GARCH effects) and
  mean-reverts. A GARCH(1,1) or EWMA weighting scheme would give more
  responsive vol estimates.

- **Normal GBM dynamics.** The Monte Carlo engine assumes geometric Brownian
  motion with constant drift and vol. Richer dynamics - stochastic
  volatility (Heston), jump-diffusion (Merton), regime switching - would
  produce more realistic tail behaviour.

- **Square-root-of-time scaling.** Multi-day VaR is derived from one-day
  VaR using the √t rule, which assumes i.i.d. returns. This breaks down
  for longer horizons where autocorrelation and mean reversion matter.

---

## Running locally

**Backend**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

**Tests**
```bash
cd backend
pytest
```

---

## Reference implementations

[`reference/`](reference/) contains standalone implementations of the core
VaR algorithms in Python, C++, R, MATLAB and SAS - each idiomatic to its
language. The Python implementation uses numpy and pandas; all others are
dependency-free. The same fixture files in
[`reference/fixtures/`](reference/fixtures/) are used to validate all
implementations. See [`reference/README.md`](reference/README.md) for
details.

---

## Infrastructure

FastAPI on AWS Lambda (arm64) behind API Gateway, with the frontend on S3 +
CloudFront. Deployed via Terraform on every push to `main`. See
`infra/bootstrap/` for the one-time setup needed before the first deploy.
