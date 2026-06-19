"""Tests for the VaR calculation engine."""

import numpy as np
import pytest

from app.var_engine import (
    backtest,
    compute_return_distribution,
    compute_return_stats,
    compute_var_surface,
    kupiec_test,
    simulate_paths,
    var_historical,
    var_monte_carlo,
    var_parametric,
)

# Synthetic returns: roughly 7.5% annual return, 24% annual vol
RNG = np.random.default_rng(42)
RETURNS = RNG.normal(0.0003, 0.015, 1000)


class TestReturnStats:
    def test_count(self):
        stats = compute_return_stats(RETURNS)
        assert stats["count"] == 1000

    def test_annualisation(self):
        stats = compute_return_stats(RETURNS)
        assert stats["std_annual"] == pytest.approx(stats["std_daily"] * np.sqrt(252), rel=0.01)

    def test_keys(self):
        stats = compute_return_stats(RETURNS)
        expected = {"count", "mean_daily", "std_daily", "mean_annual", "std_annual",
                    "skewness", "kurtosis", "min_return", "max_return"}
        assert set(stats.keys()) == expected


class TestReturnDistribution:
    def test_counts_sum(self):
        dist = compute_return_distribution(RETURNS, n_bins=30)
        assert sum(dist["counts"]) == len(RETURNS)

    def test_bin_edges_length(self):
        dist = compute_return_distribution(RETURNS, n_bins=30)
        assert len(dist["bin_edges"]) == 31  # n_bins + 1

    def test_pdf_length(self):
        dist = compute_return_distribution(RETURNS, n_bins=30)
        assert len(dist["normal_pdf"]) == 30


class TestHistoricalVar:
    def test_positive(self):
        v, cv = var_historical(RETURNS, 0.95, 1)
        assert v > 0
        assert cv > 0

    def test_cvar_geq_var(self):
        v, cv = var_historical(RETURNS, 0.95, 1)
        assert cv >= v

    def test_monotonic_confidence(self):
        v90, _ = var_historical(RETURNS, 0.90, 1)
        v95, _ = var_historical(RETURNS, 0.95, 1)
        v99, _ = var_historical(RETURNS, 0.99, 1)
        assert v90 < v95 < v99


class TestParametricVar:
    def test_positive(self):
        v, cv = var_parametric(RETURNS, 0.95, 1)
        assert v > 0
        assert cv > 0

    def test_cvar_geq_var(self):
        v, cv = var_parametric(RETURNS, 0.95, 1)
        assert cv >= v


class TestMonteCarloVar:
    def test_positive(self):
        v, cv = var_monte_carlo(RETURNS, 0.95, 1)
        assert v > 0
        assert cv > 0

    def test_cvar_geq_var(self):
        v, cv = var_monte_carlo(RETURNS, 0.95, 1)
        assert cv >= v

    def test_converges_to_parametric(self):
        r = RNG.normal(0, 0.02, 10_000)
        vm, _ = var_monte_carlo(r, 0.95, 1, n_simulations=200_000)
        vp, _ = var_parametric(r, 0.95, 1)
        assert vm == pytest.approx(vp, rel=0.05)


class TestVarSurface:
    def test_shape(self):
        surface = compute_var_surface(RETURNS, [0.90, 0.95, 0.99], [1, 5, 10])
        assert len(surface) == 9

    def test_all_positive(self):
        surface = compute_var_surface(RETURNS, [0.95], [1])
        row = surface[0]
        for key in ["var_historical", "cvar_historical", "var_parametric",
                     "cvar_parametric", "var_monte_carlo", "cvar_monte_carlo"]:
            assert row[key] > 0


class TestSimulatePaths:
    def test_shape(self):
        result = simulate_paths(RETURNS, holding_period=10, n_simulations=100)
        assert len(result["paths"]) == 100
        assert len(result["paths"][0]) == 11  # 10 steps + starting point

    def test_starts_at_one(self):
        result = simulate_paths(RETURNS, holding_period=5, n_simulations=50)
        for path in result["paths"]:
            assert path[0] == pytest.approx(1.0)


class TestBacktest:
    def test_output_keys(self):
        result = backtest(RETURNS, 0.95, 250, "historical")
        expected = {"dates", "actual_returns", "var_predictions", "breaches",
                    "breach_count", "breach_rate", "expected_breach_rate", "kupiec_p_value"}
        assert set(result.keys()) == expected

    def test_length(self):
        result = backtest(RETURNS, 0.95, 250, "historical")
        assert len(result["dates"]) == len(RETURNS) - 250

    def test_breach_count_bounded(self):
        result = backtest(RETURNS, 0.95, 250, "historical")
        assert 0 <= result["breach_count"] <= len(result["dates"])

    def test_parametric_method(self):
        result = backtest(RETURNS, 0.95, 250, "parametric")
        assert len(result["dates"]) > 0

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            backtest(RETURNS, 0.95, 250, "invalid")


class TestKupiecTest:
    def test_well_calibrated(self):
        assert kupiec_test(1000, 50, 0.95) > 0.05

    def test_miscalibrated(self):
        assert kupiec_test(1000, 120, 0.95) < 0.05
