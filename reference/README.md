# Reference implementations

Independent, side-by-side implementations of the core Value at Risk (VaR)
and Expected Shortfall (CVaR) algorithms, each idiomatic to its language:

- [`python/`](python/) - numpy + pandas
- [`cpp/`](cpp/) - C++17 (no external dependencies)
- [`r/`](r/) - base R only (`testthat` for tests only)
- [`matlab/`](matlab/) - base MATLAB, no toolboxes required
- [`sas/`](sas/) - SAS macros

Python, C++ and R are validated against the same fixture files in
[`fixtures/`](fixtures/) as part of automated CI. The MATLAB and SAS
implementations use the same fixtures but are validated manually (see their
respective READMEs) since those runtimes are not available in CI.

See the [root README](../README.md) for a full explanation of the
methodology.

## What each implementation covers

Every language implements the same set of functions:

| Function | Description |
|----------|-------------|
| `log_returns` | Convert a price series to log returns |
| `var_historical` | VaR via empirical percentile of observed returns |
| `cvar_historical` | Mean of returns beyond the VaR threshold |
| `var_parametric` | VaR under the normal distribution assumption |
| `cvar_parametric` | CVaR under the normal distribution assumption |
| `var_monte_carlo` | VaR from GBM-simulated return paths |
| `cvar_monte_carlo` | CVaR from GBM-simulated return paths |

All functions accept a confidence level (e.g. 0.95) and a holding period
in days (e.g. 1, 5, 10). Multi-day VaR is scaled from the one-day figure
using the square-root-of-time rule.

Monte Carlo functions accept a random seed parameter to ensure
deterministic, reproducible results across languages.

## Fixtures

All implementations are validated against the same fixture data in
[`fixtures/`](fixtures/):

- `fixtures/sample_returns.csv` - 1000 synthetic daily log returns drawn
  from N(0.0003, 0.015) with seed=42. Single column: `return`.

- `fixtures/expected_historical.csv` - expected Historical Simulation VaR
  and CVaR for each combination of confidence level (0.90, 0.95, 0.99)
  and holding period (1, 5, 10): `confidence,holding_period,var,cvar`.

- `fixtures/expected_parametric.csv` - expected Variance-Covariance VaR
  and CVaR, same structure.

- `fixtures/expected_monte_carlo.csv` - expected Monte Carlo VaR and CVaR
  with seed=42 and 100,000 simulations, same structure.
