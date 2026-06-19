#pragma once

#include <cmath>
#include <random>
#include <vector>

namespace var {

// ln(P_t / P_{t-1}) for each consecutive pair.
inline std::vector<double> log_returns(const std::vector<double>& prices) {
    std::vector<double> r;
    r.reserve(prices.size() - 1);
    for (size_t i = 1; i < prices.size(); ++i) {
        r.push_back(std::log(prices[i] / prices[i - 1]));
    }
    return r;
}

// Empirical percentile. Multi-day scaled by sqrt(t).
double var_historical(const std::vector<double>& returns,
                      double confidence = 0.95,
                      int holding_period = 1);

// Average loss beyond the VaR threshold.
double cvar_historical(const std::vector<double>& returns,
                       double confidence = 0.95,
                       int holding_period = 1);

// Closed-form VaR assuming N(mu, sigma^2).
double var_parametric(const std::vector<double>& returns,
                      double confidence = 0.95,
                      int holding_period = 1);

// Analytical CVaR under normality.
double cvar_parametric(const std::vector<double>& returns,
                       double confidence = 0.95,
                       int holding_period = 1);

struct MonteCarloResult {
    double var;
    double cvar;
    std::vector<double> simulated_returns;
};

// Simulate GBM paths, read VaR/CVaR off the simulated distribution.
MonteCarloResult var_monte_carlo(const std::vector<double>& returns,
                                 double confidence = 0.95,
                                 int holding_period = 1,
                                 int n_simulations = 10000,
                                 unsigned long seed = 42);

}  // namespace var
