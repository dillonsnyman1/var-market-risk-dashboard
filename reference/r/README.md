# VaR - R reference implementation

Base R only (`testthat` for tests). See [`var.R`](var.R) for the
implementation and the [root README](../../README.md) for the full
methodology description.

## Usage

```r
source("var.R")

prices <- c(100, 102, 99.5, 101.3, 98.7)
returns <- log_returns(prices)

# Or load the fixture data
returns <- read.csv("../fixtures/sample_returns.csv")$return

# Historical Simulation - 95% confidence, 1-day horizon
var_historical(returns, confidence = 0.95, holding_period = 1)
cvar_historical(returns, confidence = 0.95, holding_period = 1)

# Parametric - 99% confidence, 10-day horizon
var_parametric(returns, confidence = 0.99, holding_period = 10)

# Monte Carlo - 95% confidence, 5-day horizon
mc <- var_monte_carlo(returns, confidence = 0.95, holding_period = 5,
                      n_simulations = 100000, seed = 42)
mc$var
mc$cvar
```

## Running the tests

```bash
Rscript test_var.R
```
