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
# TODO: var_monte_carlo, cvar_monte_carlo, simulate_gbm_paths


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
# TODO: backtest_var, kupiec_test


# ---------------------------------------------------------------------------
# Surface / convenience
# ---------------------------------------------------------------------------
# TODO: compute_var_surface — all methods x confidences x holding periods
