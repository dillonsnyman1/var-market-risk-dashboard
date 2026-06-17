"""Tests for the VaR reference implementation.

Covers the methods implemented so far (Historical Simulation and
Variance-Covariance). Tests for Monte Carlo and backtesting will be
added as those functions are implemented.
"""

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
        """For N(0, sigma) returns, parametric VaR should be z_alpha * sigma."""
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
