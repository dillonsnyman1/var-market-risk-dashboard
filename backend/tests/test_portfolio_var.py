"""Tests for portfolio VaR functions."""

import numpy as np
import pytest

from app.var_engine import (
    compute_component_var,
    compute_correlation_matrix,
    compute_covariance_matrix,
    compute_incremental_var,
    compute_marginal_var,
    compute_portfolio_returns,
    compute_portfolio_var_surface,
    portfolio_var_historical,
    portfolio_var_monte_carlo,
    portfolio_var_parametric,
)

# Two synthetic assets with known properties
RNG = np.random.default_rng(42)
ASSET_A = RNG.normal(0.0003, 0.015, 500)
ASSET_B = RNG.normal(0.0001, 0.020, 500)
ASSETS = [ASSET_A, ASSET_B]
WEIGHTS = np.array([0.6, 0.4])


class TestPortfolioReturns:
    def test_length(self):
        port = compute_portfolio_returns(ASSETS, WEIGHTS)
        assert len(port) == 500

    def test_weighted_sum(self):
        port = compute_portfolio_returns(ASSETS, WEIGHTS)
        expected = 0.6 * ASSET_A + 0.4 * ASSET_B
        np.testing.assert_allclose(port, expected)

    def test_truncation(self):
        short = RNG.normal(0, 0.01, 100)
        port = compute_portfolio_returns([ASSET_A, short], np.array([0.5, 0.5]))
        assert len(port) == 100


class TestCorrelationCovariance:
    def test_correlation_shape(self):
        corr = compute_correlation_matrix(ASSETS)
        assert corr.shape == (2, 2)

    def test_correlation_diagonal(self):
        corr = compute_correlation_matrix(ASSETS)
        np.testing.assert_allclose(np.diag(corr), [1.0, 1.0], atol=1e-10)

    def test_correlation_symmetric(self):
        corr = compute_correlation_matrix(ASSETS)
        np.testing.assert_allclose(corr, corr.T)

    def test_correlation_bounded(self):
        corr = compute_correlation_matrix(ASSETS)
        assert np.all(corr >= -1.0) and np.all(corr <= 1.0)

    def test_covariance_shape(self):
        cov = compute_covariance_matrix(ASSETS)
        assert cov.shape == (2, 2)

    def test_covariance_positive_diagonal(self):
        cov = compute_covariance_matrix(ASSETS)
        assert np.all(np.diag(cov) > 0)


class TestPortfolioVarHistorical:
    def test_positive(self):
        v, cv = portfolio_var_historical(ASSETS, WEIGHTS, 0.95, 1)
        assert v > 0 and cv > 0

    def test_cvar_geq_var(self):
        v, cv = portfolio_var_historical(ASSETS, WEIGHTS, 0.95, 1)
        assert cv >= v

    def test_monotonic_confidence(self):
        v90, _ = portfolio_var_historical(ASSETS, WEIGHTS, 0.90, 1)
        v95, _ = portfolio_var_historical(ASSETS, WEIGHTS, 0.95, 1)
        v99, _ = portfolio_var_historical(ASSETS, WEIGHTS, 0.99, 1)
        assert v90 < v95 < v99


class TestPortfolioVarParametric:
    def test_positive(self):
        v, cv = portfolio_var_parametric(ASSETS, WEIGHTS, 0.95, 1)
        assert v > 0 and cv > 0

    def test_cvar_geq_var(self):
        v, cv = portfolio_var_parametric(ASSETS, WEIGHTS, 0.95, 1)
        assert cv >= v


class TestPortfolioVarMonteCarlo:
    def test_positive(self):
        v, cv = portfolio_var_monte_carlo(ASSETS, WEIGHTS, 0.95, 1)
        assert v > 0 and cv > 0

    def test_cvar_geq_var(self):
        v, cv = portfolio_var_monte_carlo(ASSETS, WEIGHTS, 0.95, 1)
        assert cv >= v

    def test_seed_reproducibility(self):
        v1, c1 = portfolio_var_monte_carlo(ASSETS, WEIGHTS, 0.95, 1, seed=99)
        v2, c2 = portfolio_var_monte_carlo(ASSETS, WEIGHTS, 0.95, 1, seed=99)
        assert v1 == v2 and c1 == c2


class TestPortfolioVarSurface:
    def test_shape(self):
        surface = compute_portfolio_var_surface(ASSETS, WEIGHTS, [0.90, 0.95, 0.99], [1, 5, 10])
        assert len(surface) == 9

    def test_all_positive(self):
        surface = compute_portfolio_var_surface(ASSETS, WEIGHTS, [0.95], [1])
        row = surface[0]
        for key in ["var_historical", "cvar_historical", "var_parametric",
                     "cvar_parametric", "var_monte_carlo", "cvar_monte_carlo"]:
            assert row[key] > 0


class TestRiskDecomposition:
    def test_component_var_sums_to_portfolio(self):
        """Component VaRs should sum to total portfolio VaR."""
        comp = compute_component_var(ASSETS, WEIGHTS, 0.95)
        port_var, _ = portfolio_var_parametric(ASSETS, WEIGHTS, 0.95, 1)
        assert sum(comp) == pytest.approx(port_var, rel=0.01)

    def test_component_var_length(self):
        comp = compute_component_var(ASSETS, WEIGHTS, 0.95)
        assert len(comp) == 2

    def test_marginal_var_length(self):
        marg = compute_marginal_var(ASSETS, WEIGHTS, 0.95)
        assert len(marg) == 2

    def test_incremental_var_length(self):
        inc = compute_incremental_var(ASSETS, WEIGHTS, 0.95)
        assert len(inc) == 2

    def test_diversification_benefit_positive(self):
        """For imperfectly correlated assets, diversification should reduce VaR."""
        standalone_sum = 0.0
        for i, r in enumerate(ASSETS):
            v, _ = portfolio_var_parametric([r], np.array([1.0]), 0.95, 1)
            standalone_sum += WEIGHTS[i] * v
        port_var, _ = portfolio_var_parametric(ASSETS, WEIGHTS, 0.95, 1)
        assert standalone_sum > port_var
