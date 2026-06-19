# VaR - MATLAB reference implementation

Base MATLAB, no toolboxes required. One function per `.m` file. See the
[root README](../../README.md) for the full methodology description.

> **Note on testing**: this implementation is validated manually against the
> shared fixtures (no MATLAB runtime in CI). Run `test_var.m` and compare
> output to the `expected_*.csv` files.

## Functions

| File | Description |
|------|-------------|
| `var_historical.m` | Empirical percentile VaR and CVaR |
| `var_parametric.m` | Closed-form VaR and CVaR under normality |
| `var_monte_carlo.m` | GBM-based Monte Carlo VaR and CVaR |

## Usage

```matlab
returns = csvread('../fixtures/sample_returns.csv', 1, 0);

% Historical - 95% confidence, 1-day
[v, cv] = var_historical(returns, 0.95, 1);

% Parametric - 99% confidence, 10-day
[v, cv] = var_parametric(returns, 0.99, 10);

% Monte Carlo - 95% confidence, 5-day, 100k sims
mc = var_monte_carlo(returns, 0.95, 5, 100000, 42);
mc.var
mc.cvar
```

## Running the tests

```matlab
cd reference/matlab
run('test_var.m')
```
