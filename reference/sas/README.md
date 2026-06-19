# VaR - SAS reference implementation

SAS macros, base SAS only. See [`var.sas`](var.sas) for the macro
definitions and the [root README](../../README.md) for the full methodology
description.

> **Note on testing**: this implementation is validated manually against the
> shared fixtures (no SAS runtime in CI). Run `test_var.sas` and compare
> the log output to the `expected_*.csv` files.

## Macros

| Macro | Description |
|-------|-------------|
| `%var_historical` | Empirical percentile VaR and CVaR |
| `%var_parametric` | Closed-form VaR and CVaR under normality |
| `%var_monte_carlo` | GBM-based Monte Carlo VaR and CVaR |

Each macro reads a dataset with a single column `return` and writes
results into caller-specified macro variables.

## Usage

```sas
%include "var.sas";

proc import datafile="sample_returns.csv" out=returns dbms=csv replace;
    getnames=yes;
run;

%var_historical(data=returns, confidence=0.95, holding_period=1,
                out_var=hist_var, out_cvar=hist_cvar);
%put VaR=&hist_var. CVaR=&hist_cvar.;

%var_parametric(data=returns, confidence=0.99, holding_period=10,
                out_var=param_var, out_cvar=param_cvar);

%var_monte_carlo(data=returns, confidence=0.95, holding_period=5,
                 n_simulations=100000, seed=42,
                 out_var=mc_var, out_cvar=mc_cvar);
```

## Running the test driver

1. Open `test_var.sas` in SAS (SAS Studio / SAS OnDemand / local install).
2. Update `fixtures_path` at the top to the absolute path of `../fixtures/`.
3. Run the script and compare the printed values to the `expected_*.csv` files.
