# VaR - Python reference implementation

numpy + pandas. See [`var.py`](var.py) for the implementation and the
[root README](../../README.md) for the full methodology description.

## Functions

| Function | Description |
|----------|-------------|
| `log_returns` | Convert a price Series to log returns |
| `var_historical` | Historical Simulation VaR |
| `cvar_historical` | Historical Simulation CVaR (Expected Shortfall) |
| `var_parametric` | Variance-Covariance VaR (normal assumption) |
| `cvar_parametric` | Variance-Covariance CVaR |
| `simulate_gbm_paths` | GBM price path simulation |
| `var_monte_carlo` | Monte Carlo VaR and CVaR via GBM simulation |
| `backtest_var` | Rolling-window backtesting with breach counting |
| `kupiec_test` | Kupiec proportion-of-failures test |
| `compute_var_surface` | All methods x confidences x holding periods |

## Usage

```python
import pandas as pd
from var import (
    log_returns,
    var_historical,
    cvar_historical,
    var_parametric,
    cvar_parametric,
    var_monte_carlo,
    compute_var_surface,
    backtest_var,
)

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

# Monte Carlo — 95% confidence, 5-day horizon
mc = var_monte_carlo(returns, confidence=0.95, holding_period=5,
                     n_simulations=100_000, seed=42)
print(mc["var"], mc["cvar"])

# Full surface: all methods x [90%, 95%, 99%] x [1, 5, 10 days]
surface = compute_var_surface(returns)
print(surface)

# Backtest: rolling 250-day window, historical method
bt = backtest_var(returns, confidence=0.95, window=250, method="historical")
print(f"Breaches: {bt.attrs['breach_count']}, Kupiec p={bt.attrs['kupiec_p_value']:.4f}")
```

## Running the example

```bash
python example.py
```

## Running the tests

```bash
pip install pytest numpy pandas scipy
pytest test_var.py -v
```
