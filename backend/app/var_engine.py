"""VaR calculation engine. Same maths as reference/python/var.py but
structured for the API layer (returns dicts/tuples instead of DataFrames)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, chi2


# ---------------------------------------------------------------------------
# Return statistics
# ---------------------------------------------------------------------------

def compute_return_stats(returns: np.ndarray) -> dict:
    """Descriptive stats for the return series. Kurtosis here is the raw
    (non-excess) fourth moment ratio - normal data gives ~3.0."""
    n = len(returns)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    m3 = ((returns - mu) ** 3).mean()
    m4 = ((returns - mu) ** 4).mean()
    return {
        "count": n,
        "mean_daily": float(mu),
        "std_daily": float(sigma),
        "mean_annual": float(mu * 252),
        "std_annual": float(sigma * np.sqrt(252)),
        "skewness": float(m3 / sigma**3),
        "kurtosis": float(m4 / sigma**4),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
    }


def compute_return_distribution(returns: np.ndarray, n_bins: int = 50) -> dict:
    """Histogram of returns with a fitted normal overlay. The PDF values are
    scaled to match the histogram counts so they can be plotted together."""
    counts, bin_edges = np.histogram(returns, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    pdf_values = norm.pdf(bin_centers, mu, sigma) * len(returns) * bin_width

    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "normal_pdf": pdf_values.tolist(),
    }


# ---------------------------------------------------------------------------
# Historical Simulation
# ---------------------------------------------------------------------------

def var_historical(returns: np.ndarray, confidence: float, holding_period: int) -> tuple[float, float]:
    var_1d = -np.percentile(returns, (1 - confidence) * 100)
    var_val = float(var_1d * np.sqrt(holding_period))

    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= threshold]
    cvar_1d = -tail.mean() if len(tail) > 0 else -threshold
    cvar_val = float(cvar_1d * np.sqrt(holding_period))

    return var_val, cvar_val


# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------

def var_parametric(returns: np.ndarray, confidence: float, holding_period: int) -> tuple[float, float]:
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(confidence)
    h = holding_period
    alpha = 1 - confidence

    var_val = float(-(mu * h - z * sigma * np.sqrt(h)))
    cvar_val = float(-(mu * h - sigma * np.sqrt(h) * norm.pdf(z) / alpha))

    return var_val, cvar_val


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def var_monte_carlo(
    returns: np.ndarray,
    confidence: float,
    holding_period: int,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    dt = holding_period
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_simulations)
    sim_returns = drift + diffusion * z

    var_val = float(-np.percentile(sim_returns, (1 - confidence) * 100))
    tail = sim_returns[sim_returns <= -var_val]
    cvar_val = float(-tail.mean()) if len(tail) > 0 else var_val

    return var_val, cvar_val


# ---------------------------------------------------------------------------
# VaR surface
# ---------------------------------------------------------------------------

def compute_var_surface(
    returns: np.ndarray,
    confidences: list[float],
    holding_periods: list[int],
    n_simulations: int = 10_000,
) -> list[dict]:
    rows = []
    for conf in confidences:
        for hp in holding_periods:
            vh, cvh = var_historical(returns, conf, hp)
            vp, cvp = var_parametric(returns, conf, hp)
            vm, cvm = var_monte_carlo(returns, conf, hp, n_simulations)
            rows.append({
                "confidence": conf,
                "holding_period": hp,
                "var_historical": vh,
                "cvar_historical": cvh,
                "var_parametric": vp,
                "cvar_parametric": cvp,
                "var_monte_carlo": vm,
                "cvar_monte_carlo": cvm,
            })
    return rows


# ---------------------------------------------------------------------------
# Monte Carlo paths (for visualisation)
# ---------------------------------------------------------------------------

def simulate_paths(
    returns: np.ndarray,
    holding_period: int,
    n_simulations: int = 500,
    seed: int = 42,
) -> dict:
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    n_steps = holding_period

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_simulations, n_steps))
    dt = 1.0
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z

    log_increments = drift + diffusion
    log_path = np.concatenate(
        [np.zeros((n_simulations, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )
    # normalised paths starting at 1.0 - frontend plots these directly
    paths = np.exp(log_path)

    terminal_returns = log_path[:, -1]
    var_95 = float(-np.percentile(terminal_returns, 5))
    var_99 = float(-np.percentile(terminal_returns, 1))

    time_points = list(range(n_steps + 1))

    return {
        "time_points": time_points,
        "paths": paths.tolist(),
        "var_95": var_95,
        "var_99": var_99,
        "terminal_returns": terminal_returns.tolist(),
    }


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def kupiec_test(n_obs: int, n_breaches: int, confidence: float) -> float:
    p = 1 - confidence
    if n_breaches == 0 or n_breaches == n_obs:
        return 1.0
    p_hat = n_breaches / n_obs
    lr = -2 * (
        n_breaches * np.log(p / p_hat)
        + (n_obs - n_breaches) * np.log((1 - p) / (1 - p_hat))
    )
    return float(1 - chi2.cdf(lr, df=1))


def backtest(
    returns: np.ndarray,
    confidence: float,
    window: int,
    method: str,
) -> dict:
    var_fn = {"historical": var_historical, "parametric": var_parametric}
    if method not in var_fn:
        raise ValueError(f"Unknown method '{method}'")

    dates = []
    actual_returns = []
    var_predictions = []
    breaches = []

    for t in range(window, len(returns)):
        trailing = returns[t - window : t]
        predicted_var, _ = var_fn[method](trailing, confidence, 1)
        actual = returns[t]
        breach = actual < -predicted_var

        dates.append(t)
        actual_returns.append(float(actual))
        var_predictions.append(float(-predicted_var))
        breaches.append(bool(breach))

    n = len(dates)
    n_breaches = sum(breaches)

    return {
        "dates": dates,
        "actual_returns": actual_returns,
        "var_predictions": var_predictions,
        "breaches": breaches,
        "breach_count": n_breaches,
        "breach_rate": n_breaches / n if n > 0 else 0.0,
        "expected_breach_rate": 1 - confidence,
        "kupiec_p_value": kupiec_test(n, n_breaches, confidence),
    }


# ---------------------------------------------------------------------------
# Portfolio VaR
# ---------------------------------------------------------------------------

def _align_returns(asset_returns: list[np.ndarray]) -> np.ndarray:
    """Truncate all series to the length of the shortest one so they line up."""
    min_len = min(len(r) for r in asset_returns)
    return np.column_stack([r[-min_len:] for r in asset_returns])


def compute_portfolio_returns(
    asset_returns: list[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    """Weighted sum of individual asset returns."""
    aligned = _align_returns(asset_returns)
    return aligned @ weights


def compute_correlation_matrix(asset_returns: list[np.ndarray]) -> np.ndarray:
    aligned = _align_returns(asset_returns)
    return np.corrcoef(aligned, rowvar=False)


def compute_covariance_matrix(asset_returns: list[np.ndarray]) -> np.ndarray:
    aligned = _align_returns(asset_returns)
    cov = np.cov(aligned, rowvar=False, ddof=1)
    # np.cov returns a scalar for a single series - force it to 2D
    return np.atleast_2d(cov)


# ---------------------------------------------------------------------------
# Portfolio VaR - three methods
# ---------------------------------------------------------------------------

def portfolio_var_historical(
    asset_returns: list[np.ndarray],
    weights: np.ndarray,
    confidence: float,
    holding_period: int,
) -> tuple[float, float]:
    """Historical VaR on the portfolio return series."""
    port_returns = compute_portfolio_returns(asset_returns, weights)
    return var_historical(port_returns, confidence, holding_period)


def portfolio_var_parametric(
    asset_returns: list[np.ndarray],
    weights: np.ndarray,
    confidence: float,
    holding_period: int,
) -> tuple[float, float]:
    """Parametric VaR using the full covariance matrix.
    sigma_p = sqrt(w^T * Sigma * w), then standard normal VaR formula.
    """
    port_returns = compute_portfolio_returns(asset_returns, weights)
    mu_p = port_returns.mean()
    cov = compute_covariance_matrix(asset_returns)
    sigma_p = float(np.sqrt(weights @ cov @ weights))

    z = norm.ppf(confidence)
    h = holding_period
    alpha = 1 - confidence

    var_val = float(-(mu_p * h - z * sigma_p * np.sqrt(h)))
    cvar_val = float(-(mu_p * h - sigma_p * np.sqrt(h) * norm.pdf(z) / alpha))
    return var_val, cvar_val


def portfolio_var_monte_carlo(
    asset_returns: list[np.ndarray],
    weights: np.ndarray,
    confidence: float,
    holding_period: int,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """MC VaR with correlated draws via Cholesky decomposition."""
    aligned = _align_returns(asset_returns)
    n_assets = aligned.shape[1]
    mu = aligned.mean(axis=0)
    cov = np.cov(aligned, rowvar=False, ddof=1)

    # Cholesky factor for generating correlated normals
    L = np.linalg.cholesky(cov)

    dt = holding_period
    drift = (mu - 0.5 * np.diag(cov)) * dt

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_simulations, n_assets))
    correlated_Z = Z @ L.T
    sim_asset_returns = drift + correlated_Z * np.sqrt(dt)

    sim_portfolio_returns = sim_asset_returns @ weights

    var_val = float(-np.percentile(sim_portfolio_returns, (1 - confidence) * 100))
    tail = sim_portfolio_returns[sim_portfolio_returns <= -var_val]
    cvar_val = float(-tail.mean()) if len(tail) > 0 else var_val

    return var_val, cvar_val


# ---------------------------------------------------------------------------
# Portfolio VaR surface
# ---------------------------------------------------------------------------

def compute_portfolio_var_surface(
    asset_returns: list[np.ndarray],
    weights: np.ndarray,
    confidences: list[float],
    holding_periods: list[int],
    n_simulations: int = 10_000,
) -> list[dict]:
    rows = []
    for conf in confidences:
        for hp in holding_periods:
            vh, cvh = portfolio_var_historical(asset_returns, weights, conf, hp)
            vp, cvp = portfolio_var_parametric(asset_returns, weights, conf, hp)
            vm, cvm = portfolio_var_monte_carlo(asset_returns, weights, conf, hp, n_simulations)
            rows.append({
                "confidence": conf,
                "holding_period": hp,
                "var_historical": vh,
                "cvar_historical": cvh,
                "var_parametric": vp,
                "cvar_parametric": cvp,
                "var_monte_carlo": vm,
                "cvar_monte_carlo": cvm,
            })
    return rows


# ---------------------------------------------------------------------------
# Risk decomposition (parametric, 1-day)
# ---------------------------------------------------------------------------

def compute_component_var(
    asset_returns: list[np.ndarray],
    weights: np.ndarray,
    confidence: float = 0.95,
) -> list[float]:
    """Component VaR for each asset. Sums to total portfolio parametric VaR.
    Uses the same mu-inclusive formula as portfolio_var_parametric so the
    numbers actually add up."""
    port_returns = compute_portfolio_returns(asset_returns, weights)
    mu_p = port_returns.mean()
    cov = compute_covariance_matrix(asset_returns)
    sigma_p = float(np.sqrt(weights @ cov @ weights))
    z = norm.ppf(confidence)

    port_var = -(mu_p - z * sigma_p)

    # each asset's share of portfolio variance, scaled to VaR
    var_contributions = weights * (cov @ weights)  # w_i * (Sigma @ w)_i
    pct_contributions = var_contributions / (sigma_p ** 2)
    return (pct_contributions * port_var).tolist()


def compute_marginal_var(
    asset_returns: list[np.ndarray],
    weights: np.ndarray,
    confidence: float = 0.95,
) -> list[float]:
    """Marginal VaR: sensitivity of portfolio VaR to a small weight increase."""
    cov = compute_covariance_matrix(asset_returns)
    sigma_p = float(np.sqrt(weights @ cov @ weights))
    z = norm.ppf(confidence)
    return ((cov @ weights) / sigma_p * z).tolist()


def compute_incremental_var(
    asset_returns: list[np.ndarray],
    weights: np.ndarray,
    confidence: float = 0.95,
) -> list[float]:
    """Incremental VaR: portfolio VaR with vs without each asset."""
    port_var, _ = portfolio_var_parametric(asset_returns, weights, confidence, 1)
    n_assets = len(asset_returns)
    result = []
    for i in range(n_assets):
        if weights[i] == 0:
            result.append(0.0)
            continue
        # remove asset i and renormalize remaining weights
        w_ex = weights.copy()
        w_ex[i] = 0.0
        w_sum = w_ex.sum()
        if w_sum > 0:
            w_ex = w_ex / w_sum
        var_ex, _ = portfolio_var_parametric(asset_returns, w_ex, confidence, 1)
        result.append(port_var - var_ex)
    return result
