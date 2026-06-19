#include "var.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace var {
namespace {

constexpr double PI = 3.14159265358979323846;

double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// Sample std dev (Bessel-corrected, n-1 denominator).
double stddev(const std::vector<double>& v) {
    double m = mean(v);
    double sum_sq = 0.0;
    for (double x : v) {
        sum_sq += (x - m) * (x - m);
    }
    return std::sqrt(sum_sq / static_cast<double>(v.size() - 1));
}

// Linear interpolation between adjacent sorted values (matches numpy default).
double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * static_cast<double>(v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// Inverse normal CDF - Acklam's rational approximation (rel error < 1.15e-9).
double norm_ppf(double p) {
    if (p <= 0.0 || p >= 1.0) {
        throw std::domain_error("norm_ppf: p must be in (0, 1)");
    }
    constexpr double a1 = -3.969683028665376e+01, a2 =  2.209460984245205e+02;
    constexpr double a3 = -2.759285104469687e+02, a4 =  1.383577518672690e+02;
    constexpr double a5 = -3.066479806614716e+01, a6 =  2.506628277459239e+00;
    constexpr double b1 = -5.447609879822406e+01, b2 =  1.615858368580409e+02;
    constexpr double b3 = -1.556989798598866e+02, b4 =  6.680131188771972e+01;
    constexpr double b5 = -1.328068155288572e+01;
    constexpr double c1 = -7.784894002430293e-03, c2 = -3.223964580411365e-01;
    constexpr double c3 = -2.400758277161838e+00, c4 = -2.549732539343734e+00;
    constexpr double c5 =  4.374664141464968e+00, c6 =  2.938163982698783e+00;
    constexpr double d1 =  7.784695709041462e-03, d2 =  3.224671290700398e-01;
    constexpr double d3 =  2.445134137142996e+00, d4 =  3.754408661907416e+00;
    constexpr double p_low = 0.02425, p_high = 1.0 - p_low;
    double q, r;
    if (p < p_low) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
    } else if (p <= p_high) {
        q = p - 0.5;
        r = q * q;
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
               (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                 ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
    }
}

// N(0,1) density.
double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * PI);
}

}  // namespace

// ---------------------------------------------------------------------------
// Historical Simulation
// ---------------------------------------------------------------------------

double var_historical(const std::vector<double>& returns,
                      double confidence, int holding_period) {
    double var_1d = -percentile(returns, (1.0 - confidence) * 100.0);
    return var_1d * std::sqrt(static_cast<double>(holding_period));
}

double cvar_historical(const std::vector<double>& returns,
                       double confidence, int holding_period) {
    double threshold = percentile(returns, (1.0 - confidence) * 100.0);
    double sum = 0.0;
    int count = 0;
    for (double r : returns) {
        if (r <= threshold) {
            sum += r;
            ++count;
        }
    }
    double cvar_1d = (count > 0) ? -(sum / count) : -threshold;
    return cvar_1d * std::sqrt(static_cast<double>(holding_period));
}

// ---------------------------------------------------------------------------
// Variance-Covariance (Parametric)
// ---------------------------------------------------------------------------

double var_parametric(const std::vector<double>& returns,
                      double confidence, int holding_period) {
    double mu = mean(returns);
    double sigma = stddev(returns);
    double z = norm_ppf(confidence);
    double h = static_cast<double>(holding_period);
    return -(mu * h - z * sigma * std::sqrt(h));
}

double cvar_parametric(const std::vector<double>& returns,
                       double confidence, int holding_period) {
    double mu = mean(returns);
    double sigma = stddev(returns);
    double z = norm_ppf(confidence);
    double alpha = 1.0 - confidence;
    double h = static_cast<double>(holding_period);
    return -(mu * h - sigma * std::sqrt(h) * norm_pdf(z) / alpha);
}

// ---------------------------------------------------------------------------
// Monte Carlo Simulation
// ---------------------------------------------------------------------------

MonteCarloResult var_monte_carlo(const std::vector<double>& returns,
                                  double confidence, int holding_period,
                                  int n_simulations, unsigned long seed) {
    double mu = mean(returns);
    double sigma = stddev(returns);
    double dt = static_cast<double>(holding_period);
    double drift = (mu - 0.5 * sigma * sigma) * dt;
    double diffusion = sigma * std::sqrt(dt);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> sim_returns(n_simulations);
    for (int i = 0; i < n_simulations; ++i) {
        double z = dist(rng);
        sim_returns[i] = drift + diffusion * z;
    }

    double var_val = -percentile(sim_returns, (1.0 - confidence) * 100.0);

    double tail_sum = 0.0;
    int tail_count = 0;
    for (double r : sim_returns) {
        if (r <= -var_val) {
            tail_sum += r;
            ++tail_count;
        }
    }
    double cvar_val = (tail_count > 0) ? -(tail_sum / tail_count) : var_val;

    return MonteCarloResult{var_val, cvar_val, std::move(sim_returns)};
}

}  // namespace var
