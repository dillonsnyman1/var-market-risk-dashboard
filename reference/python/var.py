"""
Reference implementation of VaR and Expected Shortfall (CVaR).

Three methods side-by-side - Historical Simulation, Variance-Covariance,
Monte Carlo - with rolling-window backtesting and the Kupiec POF test.
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
    """Sort the returns, read off the percentile. Simple and assumption-free,
    but only as good as the history you feed it.

    Multi-day scaling uses the sqrt(t) rule, which assumes i.i.d. returns -
    not great over long horizons, but standard practice.
    """
    r = np.asarray(returns, dtype=float)
    var_1d = -np.percentile(r, (1 - confidence) * 100)
    return float(var_1d * np.sqrt(holding_period))


def cvar_historical(
    returns: np.ndarray,
    confidence: float = 0.95,
    holding_period: int = 1,
) -> float:
    """Average loss in the tail beyond VaR. Always >= VaR."""
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
    """Closed-form VaR assuming returns ~ N(mu, sigma^2).

    Fast and transparent, but the normal assumption systematically
    underestimates tail risk - real equity returns have fat tails and
    negative skew that this ignores.
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
    """Analytical CVaR under normality. Uses the standard result:
    CVaR = -(mu*h - sigma*sqrt(h) * phi(z_alpha) / (1 - alpha)).
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
    """Generate GBM price paths. Returns array of shape (n_simulations, n_steps + 1)
    where column 0 is S0. Defaults to one step per day if n_steps is omitted.
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
    """Estimate mu/sigma from the historical series, simulate GBM paths,
    then read VaR and CVaR off the simulated distribution.

    Returns dict with var, cvar, and the raw simulated_returns array
    (useful for plotting terminal distributions).
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
    """Kupiec POF test - checks whether the observed breach count is
    consistent with the expected rate (1 - confidence).

    LR statistic is chi-squared(1) under H0. Returns the p-value;
    low p-value means the model is producing too many or too few breaches.
    """
    p = 1 - confidence
    p_hat = n_breaches / n_observations if n_observations > 0 else 0.0
    n = n_observations
    x = n_breaches

    if x == 0 or x == n:
        return 1.0  # can't take log(0), and we can't reject anyway

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
    """Walk forward through the series: at each step, estimate VaR from
    the trailing window and check if the next-day return breaches it.

    Returns a DataFrame of predictions vs actuals. Summary stats
    (breach_count, breach_rate, kupiec_p_value) are in df.attrs.
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

def compute_var_surface(
    returns: np.ndarray,
    confidences: list[float] | None = None,
    holding_periods: list[int] | None = None,
    n_simulations: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all three methods across every confidence/horizon combination.
    Returns a tidy DataFrame - one row per (confidence, holding_period) pair.
    """
    if confidences is None:
        confidences = [0.90, 0.95, 0.99]
    if holding_periods is None:
        holding_periods = [1, 5, 10]

    r = np.asarray(returns, dtype=float)
    rows = []

    for conf in confidences:
        for hp in holding_periods:
            mc = var_monte_carlo(r, confidence=conf, holding_period=hp,
                                n_simulations=n_simulations, seed=seed)
            rows.append({
                "confidence": conf,
                "holding_period": hp,
                "var_historical": var_historical(r, conf, hp),
                "cvar_historical": cvar_historical(r, conf, hp),
                "var_parametric": var_parametric(r, conf, hp),
                "cvar_parametric": cvar_parametric(r, conf, hp),
                "var_monte_carlo": mc["var"],
                "cvar_monte_carlo": mc["cvar"],
            })

    return pd.DataFrame(rows)
