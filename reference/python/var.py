"""
Value at Risk (VaR) and Expected Shortfall (CVaR) reference implementation.

Implements three VaR estimation methods — Historical Simulation,
Variance-Covariance (Parametric), and Monte Carlo — plus rolling-window
backtesting with the Kupiec proportion-of-failures test.

Dependencies: numpy, pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_returns(prices: pd.Series) -> pd.Series:
    """Convert a price series to log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


# ---------------------------------------------------------------------------
# Historical Simulation
# ---------------------------------------------------------------------------

def var_historical(
    returns: np.ndarray,
    confidence: float = 0.95,
    holding_period: int = 1,
) -> float:
    """VaR via the empirical percentile of observed returns.

    Multi-day VaR is scaled from the one-day figure using the
    square-root-of-time rule (assumes i.i.d. returns).
    """
    r = np.asarray(returns, dtype=float)
    var_1d = -np.percentile(r, (1 - confidence) * 100)
    return float(var_1d * np.sqrt(holding_period))


def cvar_historical(
    returns: np.ndarray,
    confidence: float = 0.95,
    holding_period: int = 1,
) -> float:
    """Expected Shortfall (CVaR) — mean of returns beyond the VaR threshold."""
    r = np.asarray(returns, dtype=float)
    threshold = np.percentile(r, (1 - confidence) * 100)
    tail = r[r <= threshold]
    cvar_1d = -tail.mean() if len(tail) > 0 else -threshold
    return float(cvar_1d * np.sqrt(holding_period))


# ---------------------------------------------------------------------------
# Variance-Covariance (Parametric)
# ---------------------------------------------------------------------------

def var_parametric(
    returns: np.ndarray,
    confidence: float = 0.95,
    holding_period: int = 1,
) -> float:
    """VaR under the normal distribution assumption.

    VaR = -(mu * h - z_alpha * sigma * sqrt(h))

    The normality assumption underestimates tail risk when returns exhibit
    fat tails (excess kurtosis) or negative skew, which is typical for
    equity returns.
    """
    r = np.asarray(returns, dtype=float)
    mu = r.mean()
    sigma = r.std(ddof=1)
    z = norm.ppf(confidence)
    return float(-(mu * holding_period - z * sigma * np.sqrt(holding_period)))


def cvar_parametric(
    returns: np.ndarray,
    confidence: float = 0.95,
    holding_period: int = 1,
) -> float:
    """CVaR under the normal distribution assumption.

    CVaR = -(mu * h - sigma * sqrt(h) * phi(z_alpha) / (1 - alpha))

    where phi is the standard normal PDF and z_alpha = Phi^{-1}(alpha).
    """
    r = np.asarray(returns, dtype=float)
    mu = r.mean()
    sigma = r.std(ddof=1)
    z = norm.ppf(confidence)
    alpha = 1 - confidence
    return float(-(mu * holding_period - sigma * np.sqrt(holding_period) * norm.pdf(z) / alpha))


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    holding_period: int,
    n_simulations: int = 10_000,
    n_steps: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Simulate price paths via Geometric Brownian Motion.

    S_t = S_0 * exp((mu - sigma^2/2)*t + sigma*sqrt(t)*Z)

    Returns a 2D array of shape (n_simulations, n_steps + 1) where
    column 0 is S0 and the final column is the terminal price.
    If n_steps is None it defaults to holding_period (one step per day).
    """
    if n_steps is None:
        n_steps = holding_period
    dt = holding_period / n_steps
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_simulations, n_steps))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_increments = drift + diffusion
    log_path = np.concatenate(
        [np.zeros((n_simulations, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )
    return S0 * np.exp(log_path)


def var_monte_carlo(
    returns: np.ndarray,
    confidence: float = 0.95,
    holding_period: int = 1,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict:
    """VaR and CVaR from Monte Carlo simulation of GBM paths.

    Estimates mu and sigma from the historical return series, simulates
    n_simulations price paths over the holding period, and computes VaR
    and CVaR from the distribution of simulated returns.

    Returns a dict with keys: var, cvar, simulated_returns.
    """
    r = np.asarray(returns, dtype=float)
    mu = r.mean()
    sigma = r.std(ddof=1)

    paths = simulate_gbm_paths(
        S0=1.0,
        mu=mu,
        sigma=sigma,
        holding_period=holding_period,
        n_simulations=n_simulations,
        seed=seed,
    )
    simulated_returns = np.log(paths[:, -1] / paths[:, 0])

    var = float(-np.percentile(simulated_returns, (1 - confidence) * 100))
    tail = simulated_returns[simulated_returns <= -var]
    cvar = float(-tail.mean()) if len(tail) > 0 else var

    return {
        "var": var,
        "cvar": cvar,
        "simulated_returns": simulated_returns,
    }


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def kupiec_test(n_observations: int, n_breaches: int, confidence: float) -> float:
    """Kupiec proportion-of-failures (POF) likelihood-ratio test.

    Tests whether the observed breach rate is consistent with the expected
    rate (1 - confidence). Under H0 (model is correctly calibrated) the
    test statistic follows chi-squared(1).

    Returns the p-value.
    """
    p = 1 - confidence
    p_hat = n_breaches / n_observations if n_observations > 0 else 0.0
    n = n_observations
    x = n_breaches

    if x == 0 or x == n:
        # Edge case: log(0) — return p=1 (cannot reject)
        return 1.0

    lr = -2 * (
        x * np.log(p / p_hat) + (n - x) * np.log((1 - p) / (1 - p_hat))
    )
    from scipy.stats import chi2
    return float(1 - chi2.cdf(lr, df=1))


def backtest_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    window: int = 250,
    method: str = "historical",
) -> pd.DataFrame:
    """Rolling-window VaR backtest.

    At each date t (starting from index `window`), VaR is estimated from
    the trailing `window` observations and compared against the actual
    return at t. A breach occurs when the actual loss exceeds the
    predicted VaR.

    Returns a DataFrame with columns: actual_return, var_prediction,
    breach. Also attaches summary attributes: breach_count, breach_rate,
    expected_breach_rate, kupiec_p_value.
    """
    r = np.asarray(returns, dtype=float)
    var_fn = {
        "historical": var_historical,
        "parametric": var_parametric,
    }
    if method not in var_fn:
        raise ValueError(f"Unknown method '{method}'. Use 'historical' or 'parametric'.")

    records = []
    for t in range(window, len(r)):
        trailing = r[t - window : t]
        predicted_var = var_fn[method](trailing, confidence=confidence)
        actual = r[t]
        breach = actual < -predicted_var
        records.append({
            "date_index": t,
            "actual_return": float(actual),
            "var_prediction": float(-predicted_var),
            "breach": bool(breach),
        })

    df = pd.DataFrame(records)
    n = len(df)
    n_breaches = int(df["breach"].sum())

    df.attrs["breach_count"] = n_breaches
    df.attrs["breach_rate"] = n_breaches / n if n > 0 else 0.0
    df.attrs["expected_breach_rate"] = 1 - confidence
    df.attrs["kupiec_p_value"] = kupiec_test(n, n_breaches, confidence)

    return df


# ---------------------------------------------------------------------------
# Surface / convenience
# ---------------------------------------------------------------------------
# TODO: compute_var_surface — all methods x confidences x holding periods
