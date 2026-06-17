# VaR - Python reference implementation

numpy + pandas. See [`var.py`](var.py) for the implementation and the
[root README](../../README.md) for the full methodology description.

## Status

**Implemented:**
- `log_returns` — price series to log returns
- `var_historical` — Historical Simulation VaR
- `cvar_historical` — Historical Simulation CVaR (Expected Shortfall)
- `var_parametric` — Variance-Covariance VaR (normal assumption)
- `cvar_parametric` — Variance-Covariance CVaR

**Still to do:**
- `var_monte_carlo` / `cvar_monte_carlo` — Monte Carlo VaR via GBM
  simulation
- `simulate_gbm_paths` — generate simulated price paths for
  visualisation
- `backtest_var` — rolling-window backtesting with breach counting
- `kupiec_test` — Kupiec proportion-of-failures test
- `compute_var_surface` — all methods x confidences x holding periods
- `example.py` — usage demo script
- Expected output fixtures (`expected_historical.csv`, etc.)

## Usage

```python
import numpy as np
import pandas as pd
from var import log_returns, var_historical, cvar_historical, var_parametric, cvar_parametric

# From a price series
prices = pd.Series([100.0, 102.0, 99.5, 101.3, 98.7, ...])
returns = log_returns(prices)

# Or load the fixture data directly
returns = pd.read_csv("../fixtures/sample_returns.csv")["return"].values

# Historical Simulation — 95% confidence, 1-day horizon
var_hist = var_historical(returns, confidence=0.95, holding_period=1)
cvar_hist = cvar_historical(returns, confidence=0.95, holding_period=1)

# Parametric (Variance-Covariance) — 99% confidence, 10-day horizon
var_param = var_parametric(returns, confidence=0.99, holding_period=10)
cvar_param = cvar_parametric(returns, confidence=0.99, holding_period=10)
```

## Running the tests

```bash
pip install pytest numpy pandas scipy
pytest test_var.py -v
```
