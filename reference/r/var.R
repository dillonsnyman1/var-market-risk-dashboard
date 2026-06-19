# VaR and Expected Shortfall (CVaR) reference implementation.
#
# Three methods - Historical Simulation, Variance-Covariance, Monte Carlo -
# plus the Kupiec POF test. Base R only, no packages required.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log_returns <- function(prices) {
  diff(log(prices))
}

# ---------------------------------------------------------------------------
# Historical Simulation
# ---------------------------------------------------------------------------

# Empirical percentile. Multi-day scaled by sqrt(t).
var_historical <- function(returns, confidence = 0.95, holding_period = 1) {
  var_1d <- -quantile(returns, 1 - confidence, names = FALSE)
  var_1d * sqrt(holding_period)
}

# Average loss beyond the VaR threshold.
cvar_historical <- function(returns, confidence = 0.95, holding_period = 1) {
  threshold <- quantile(returns, 1 - confidence, names = FALSE)
  tail_returns <- returns[returns <= threshold]
  cvar_1d <- if (length(tail_returns) > 0) -mean(tail_returns) else -threshold
  cvar_1d * sqrt(holding_period)
}

# ---------------------------------------------------------------------------
# Variance-Covariance (Parametric)
# ---------------------------------------------------------------------------

# Closed-form VaR assuming N(mu, sigma^2).
var_parametric <- function(returns, confidence = 0.95, holding_period = 1) {
  mu <- mean(returns)
  sigma <- sd(returns)
  z <- qnorm(confidence)
  -(mu * holding_period - z * sigma * sqrt(holding_period))
}

# Analytical CVaR under normality.
cvar_parametric <- function(returns, confidence = 0.95, holding_period = 1) {
  mu <- mean(returns)
  sigma <- sd(returns)
  z <- qnorm(confidence)
  alpha <- 1 - confidence
  -(mu * holding_period - sigma * sqrt(holding_period) * dnorm(z) / alpha)
}

# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

# Simulate GBM paths, read VaR/CVaR off the simulated distribution.
# Returns a list with var, cvar, and the simulated_returns vector.
var_monte_carlo <- function(returns, confidence = 0.95, holding_period = 1,
                            n_simulations = 10000, seed = 42) {
  mu <- mean(returns)
  sigma <- sd(returns)
  dt <- holding_period
  drift <- (mu - 0.5 * sigma^2) * dt
  diffusion <- sigma * sqrt(dt)

  set.seed(seed)
  z <- rnorm(n_simulations)
  sim_returns <- drift + diffusion * z

  var_val <- -quantile(sim_returns, 1 - confidence, names = FALSE)
  tail_returns <- sim_returns[sim_returns <= -var_val]
  cvar_val <- if (length(tail_returns) > 0) -mean(tail_returns) else var_val

  list(var = var_val, cvar = cvar_val, simulated_returns = sim_returns)
}
