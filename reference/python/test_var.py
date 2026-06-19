"""Tests for the VaR reference implementation."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from var import (
    log_returns,
    var_historical,
    cvar_historical,
    var_parametric,
    cvar_parametric,
    simulate_gbm_paths,
    var_monte_carlo,
    kupiec_test,
    backtest_var,
    compute_var_surface,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_returns() -> np.ndarray:
    df = pd.read_csv(os.path.join(FIXTURES, "sample_returns.csv"))
    return df["return"].values


# ---------------------------------------------------------------------------
# log_returns
# ---------------------------------------------------------------------------

class TestLogReturns:
    def test_length(self):
        prices = pd.Series([100.0, 105.0, 103.0, 107.0])
        lr = log_returns(prices)
        assert len(lr) == len(prices) - 1

    def test_values(self):
        prices = pd.Series([100.0, 110.0])
        lr = log_returns(prices)
        assert lr.iloc[0] == pytest.approx(np.log(1.1))

    def test_round_trip(self):
        prices = pd.Series([100.0, 105.0, 103.0, 107.0, 102.0])
        lr = log_returns(prices)
        reconstructed = prices.iloc[0] * np.exp(lr.cumsum())
        np.testing.assert_allclose(reconstructed.values, prices.iloc[1:].values)


# ---------------------------------------------------------------------------
# Historical Simulation
# ---------------------------------------------------------------------------

class TestHistoricalVar:
    def test_positive(self, sample_returns):
        assert var_historical(sample_returns) > 0

    def test_higher_confidence_higher_var(self, sample_returns):
        v90 = var_historical(sample_returns, confidence=0.90)
        v95 = var_historical(sample_returns, confidence=0.95)
        v99 = var_historical(sample_returns, confidence=0.99)
        assert v90 < v95 < v99

    def test_longer_horizon_higher_var(self, sample_returns):
        v1 = var_historical(sample_returns, holding_period=1)
        v5 = var_historical(sample_returns, holding_period=5)
        v10 = var_historical(sample_returns, holding_period=10)
        assert v1 < v5 < v10

    def test_sqrt_scaling(self, sample_returns):
        v1 = var_historical(sample_returns, holding_period=1)
        v5 = var_historical(sample_returns, holding_period=5)
        assert v5 == pytest.approx(v1 * np.sqrt(5))


class TestHistoricalCvar:
    def test_positive(self, sample_returns):
        assert cvar_historical(sample_returns) > 0

    def test_cvar_geq_var(self, sample_returns):
        for conf in [0.90, 0.95, 0.99]:
            v = var_historical(sample_returns, confidence=conf)
            cv = cvar_historical(sample_returns, confidence=conf)
            assert cv >= v, f"CVaR ({cv}) < VaR ({v}) at {conf}"


# ---------------------------------------------------------------------------
# Parametric (Variance-Covariance)
# ---------------------------------------------------------------------------

class TestParametricVar:
    def test_positive(self, sample_returns):
        assert var_parametric(sample_returns) > 0

    def test_higher_confidence_higher_var(self, sample_returns):
        v90 = var_parametric(sample_returns, confidence=0.90)
        v95 = var_parametric(sample_returns, confidence=0.95)
        v99 = var_parametric(sample_returns, confidence=0.99)
        assert v90 < v95 < v99

    def test_longer_horizon_higher_var(self, sample_returns):
        v1 = var_parametric(sample_returns, holding_period=1)
        v5 = var_parametric(sample_returns, holding_period=5)
        v10 = var_parametric(sample_returns, holding_period=10)
        assert v1 < v5 < v10

    def test_known_normal(self):
        """Sanity check: for zero-mean normal data, VaR should be z * sigma."""
        rng = np.random.default_rng(99)
        r = rng.normal(0, 0.02, 100_000)
        v = var_parametric(r, confidence=0.95, holding_period=1)
        expected = 1.6449 * 0.02
        assert v == pytest.approx(expected, rel=0.02)


class TestParametricCvar:
    def test_positive(self, sample_returns):
        assert cvar_parametric(sample_returns) > 0

    def test_cvar_geq_var(self, sample_returns):
        for conf in [0.90, 0.95, 0.99]:
            v = var_parametric(sample_returns, confidence=conf)
            cv = cvar_parametric(sample_returns, confidence=conf)
            assert cv >= v, f"CVaR ({cv}) < VaR ({v}) at {conf}"


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

class TestSimulateGbmPaths:
    def test_shape(self):
        paths = simulate_gbm_paths(S0=100.0, mu=0.0, sigma=0.2, holding_period=10, n_simulations=500)
        assert paths.shape == (500, 11)

    def test_starts_at_s0(self):
        paths = simulate_gbm_paths(S0=42.0, mu=0.05, sigma=0.15, holding_period=5, n_simulations=100)
        np.testing.assert_allclose(paths[:, 0], 42.0)

    def test_seed_reproducibility(self):
        p1 = simulate_gbm_paths(S0=100.0, mu=0.0, sigma=0.2, holding_period=10, seed=7)
        p2 = simulate_gbm_paths(S0=100.0, mu=0.0, sigma=0.2, holding_period=10, seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_all_positive(self):
        paths = simulate_gbm_paths(S0=100.0, mu=-0.1, sigma=0.5, holding_period=20, n_simulations=1000)
        assert np.all(paths > 0)

    def test_custom_steps(self):
        paths = simulate_gbm_paths(S0=100.0, mu=0.0, sigma=0.2, holding_period=5, n_simulations=100, n_steps=25)
        assert paths.shape == (100, 26)


class TestMonteCarloVar:
    def test_positive(self, sample_returns):
        result = var_monte_carlo(sample_returns)
        assert result["var"] > 0

    def test_cvar_geq_var(self, sample_returns):
        result = var_monte_carlo(sample_returns, confidence=0.95)
        assert result["cvar"] >= result["var"]

    def test_seed_reproducibility(self, sample_returns):
        r1 = var_monte_carlo(sample_returns, seed=123)
        r2 = var_monte_carlo(sample_returns, seed=123)
        assert r1["var"] == r2["var"]
        assert r1["cvar"] == r2["cvar"]

    def test_simulated_returns_length(self, sample_returns):
        n = 5_000
        result = var_monte_carlo(sample_returns, n_simulations=n)
        assert len(result["simulated_returns"]) == n

    def test_converges_to_parametric(self):
        """MC should land close to parametric when the data is actually normal."""
        rng = np.random.default_rng(99)
        r = rng.normal(0, 0.02, 10_000)
        mc = var_monte_carlo(r, confidence=0.95, n_simulations=200_000, seed=42)
        param = var_parametric(r, confidence=0.95)
        assert mc["var"] == pytest.approx(param, rel=0.05)

    def test_higher_confidence_higher_var(self, sample_returns):
        v90 = var_monte_carlo(sample_returns, confidence=0.90)["var"]
        v95 = var_monte_carlo(sample_returns, confidence=0.95)["var"]
        v99 = var_monte_carlo(sample_returns, confidence=0.99)["var"]
        assert v90 < v95 < v99


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

class TestKupiecTest:
    def test_well_calibrated(self):
        """50 breaches out of 1000 at 95% - should not reject."""
        p_value = kupiec_test(n_observations=1000, n_breaches=50, confidence=0.95)
        assert p_value > 0.05

    def test_too_many_breaches(self):
        """120 breaches out of 1000 at 95% - clearly miscalibrated."""
        p_value = kupiec_test(n_observations=1000, n_breaches=120, confidence=0.95)
        assert p_value < 0.05

    def test_edge_zero_breaches(self):
        p_value = kupiec_test(n_observations=1000, n_breaches=0, confidence=0.95)
        assert p_value == 1.0

    def test_edge_all_breaches(self):
        p_value = kupiec_test(n_observations=100, n_breaches=100, confidence=0.95)
        assert p_value == 1.0


class TestBacktestVar:
    def test_output_shape(self, sample_returns):
        df = backtest_var(sample_returns, window=250)
        assert len(df) == len(sample_returns) - 250
        assert set(df.columns) == {"date_index", "actual_return", "var_prediction", "breach"}

    def test_breach_count_within_bounds(self, sample_returns):
        df = backtest_var(sample_returns, confidence=0.95, window=250)
        assert 0 <= df.attrs["breach_count"] <= len(df)

    def test_breach_rate_reasonable(self, sample_returns):
        """Synthetic data is well-behaved, so breach rate shouldn't be wild."""
        df = backtest_var(sample_returns, confidence=0.95, window=250)
        assert 0.0 < df.attrs["breach_rate"] < 0.20

    def test_expected_breach_rate(self, sample_returns):
        df = backtest_var(sample_returns, confidence=0.95)
        assert df.attrs["expected_breach_rate"] == pytest.approx(0.05)

    def test_kupiec_attached(self, sample_returns):
        df = backtest_var(sample_returns, confidence=0.95)
        assert 0.0 <= df.attrs["kupiec_p_value"] <= 1.0

    def test_historical_method(self, sample_returns):
        df = backtest_var(sample_returns, method="historical")
        assert len(df) > 0

    def test_parametric_method(self, sample_returns):
        df = backtest_var(sample_returns, method="parametric")
        assert len(df) > 0

    def test_invalid_method(self, sample_returns):
        with pytest.raises(ValueError, match="Unknown method"):
            backtest_var(sample_returns, method="invalid")


# ---------------------------------------------------------------------------
# VaR Surface
# ---------------------------------------------------------------------------

class TestVarSurface:
    def test_shape(self, sample_returns):
        df = compute_var_surface(sample_returns)
        assert len(df) == 9  # 3 confidences x 3 holding periods

    def test_columns(self, sample_returns):
        df = compute_var_surface(sample_returns)
        expected_cols = {
            "confidence", "holding_period",
            "var_historical", "cvar_historical",
            "var_parametric", "cvar_parametric",
            "var_monte_carlo", "cvar_monte_carlo",
        }
        assert set(df.columns) == expected_cols

    def test_all_positive(self, sample_returns):
        df = compute_var_surface(sample_returns)
        value_cols = [c for c in df.columns if c.startswith("var_") or c.startswith("cvar_")]
        for col in value_cols:
            assert (df[col] > 0).all(), f"{col} contains non-positive values"

    def test_cvar_geq_var(self, sample_returns):
        df = compute_var_surface(sample_returns)
        for method in ["historical", "parametric", "monte_carlo"]:
            assert (df[f"cvar_{method}"] >= df[f"var_{method}"]).all()

    def test_monotonic_in_confidence(self, sample_returns):
        df = compute_var_surface(sample_returns)
        for hp in [1, 5, 10]:
            subset = df[df["holding_period"] == hp].sort_values("confidence")
            for method in ["historical", "parametric", "monte_carlo"]:
                vals = subset[f"var_{method}"].values
                assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_monotonic_in_holding_period(self, sample_returns):
        df = compute_var_surface(sample_returns)
        for conf in [0.90, 0.95, 0.99]:
            subset = df[df["confidence"] == conf].sort_values("holding_period")
            for method in ["historical", "parametric", "monte_carlo"]:
                vals = subset[f"var_{method}"].values
                assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_custom_params(self, sample_returns):
        df = compute_var_surface(sample_returns, confidences=[0.95], holding_periods=[1, 5])
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Fixture validation
# ---------------------------------------------------------------------------

class TestFixtureValidation:
    """Check we match the locked-down fixture values exactly."""

    def test_historical_fixture(self, sample_returns):
        expected = pd.read_csv(os.path.join(FIXTURES, "expected_historical.csv"))
        for _, row in expected.iterrows():
            v = var_historical(sample_returns, row["confidence"], int(row["holding_period"]))
            cv = cvar_historical(sample_returns, row["confidence"], int(row["holding_period"]))
            assert v == pytest.approx(row["var"], rel=1e-6), \
                f"Historical VaR mismatch at conf={row['confidence']}, hp={row['holding_period']}"
            assert cv == pytest.approx(row["cvar"], rel=1e-6), \
                f"Historical CVaR mismatch at conf={row['confidence']}, hp={row['holding_period']}"

    def test_parametric_fixture(self, sample_returns):
        expected = pd.read_csv(os.path.join(FIXTURES, "expected_parametric.csv"))
        for _, row in expected.iterrows():
            v = var_parametric(sample_returns, row["confidence"], int(row["holding_period"]))
            cv = cvar_parametric(sample_returns, row["confidence"], int(row["holding_period"]))
            assert v == pytest.approx(row["var"], rel=1e-6), \
                f"Parametric VaR mismatch at conf={row['confidence']}, hp={row['holding_period']}"
            assert cv == pytest.approx(row["cvar"], rel=1e-6), \
                f"Parametric CVaR mismatch at conf={row['confidence']}, hp={row['holding_period']}"

    def test_monte_carlo_fixture(self, sample_returns):
        expected = pd.read_csv(os.path.join(FIXTURES, "expected_monte_carlo.csv"))
        for _, row in expected.iterrows():
            mc = var_monte_carlo(
                sample_returns, row["confidence"], int(row["holding_period"]),
                n_simulations=100_000, seed=42,
            )
            assert mc["var"] == pytest.approx(row["var"], rel=1e-6), \
                f"MC VaR mismatch at conf={row['confidence']}, hp={row['holding_period']}"
            assert mc["cvar"] == pytest.approx(row["cvar"], rel=1e-6), \
                f"MC CVaR mismatch at conf={row['confidence']}, hp={row['holding_period']}"
