# VaR - C++ reference implementation

C++17, no external dependencies. See [`var.hpp`](var.hpp) /
[`var.cpp`](var.cpp) for the implementation and the
[root README](../../README.md) for the full methodology description.

## Usage

```cpp
#include "var.hpp"

#include <vector>

// From a price series
std::vector<double> prices = {100.0, 102.0, 99.5, 101.3, 98.7};
std::vector<double> returns = var::log_returns(prices);

// Historical Simulation - 95% confidence, 1-day horizon
double v = var::var_historical(returns, 0.95, 1);
double cv = var::cvar_historical(returns, 0.95, 1);

// Parametric (Variance-Covariance) - 99% confidence, 10-day horizon
double vp = var::var_parametric(returns, 0.99, 10);
double cvp = var::cvar_parametric(returns, 0.99, 10);

// Monte Carlo - 95% confidence, 5-day horizon, 100k sims
var::MonteCarloResult mc = var::var_monte_carlo(returns, 0.95, 5, 100000, 42);
// mc.var, mc.cvar, mc.simulated_returns
```

## Building and running the tests

```bash
cmake -S . -B build
cmake --build build
./build/test_var
```

Or via CTest:

```bash
ctest --test-dir build
```
