#include "var.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Bare-bones CSV reader - just enough for our fixture format.
std::vector<std::vector<std::string>> read_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open fixture file: " + path);
    }
    std::vector<std::vector<std::string>> rows;
    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (first) { first = false; continue; }
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }
        rows.push_back(fields);
    }
    return rows;
}

std::vector<double> load_returns(const std::string& path) {
    std::vector<double> returns;
    for (const auto& row : read_csv(path)) {
        returns.push_back(std::stod(row[0]));
    }
    return returns;
}

struct ExpectedRow {
    double confidence;
    int holding_period;
    double var;
    double cvar;
};

std::vector<ExpectedRow> load_expected(const std::string& path) {
    std::vector<ExpectedRow> rows;
    for (const auto& row : read_csv(path)) {
        rows.push_back({std::stod(row[0]), std::stoi(row[1]),
                         std::stod(row[2]), std::stod(row[3])});
    }
    return rows;
}

void assert_approx(double actual, double expected, double rel_tol,
                   const std::string& label) {
    double diff = std::abs(actual - expected);
    double threshold = rel_tol * std::abs(expected);
    if (diff > threshold && diff > 1e-10) {
        std::cerr << "FAIL: " << label
                  << " - expected " << expected
                  << ", got " << actual
                  << " (diff " << diff << ")" << std::endl;
        assert(false);
    }
}

std::string fixtures_dir() {
#ifdef FIXTURES_DIR
    return FIXTURES_DIR;
#else
    return "../fixtures";
#endif
}

int tests_run = 0;
int tests_passed = 0;

void pass(const std::string& name) {
    ++tests_run;
    ++tests_passed;
    std::cout << "  PASS  " << name << std::endl;
}

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_log_returns() {
    std::vector<double> prices = {100.0, 105.0, 103.0, 107.0};
    auto lr = var::log_returns(prices);
    assert(lr.size() == 3);
    assert_approx(lr[0], std::log(105.0 / 100.0), 1e-10, "log_returns[0]");
    pass("log_returns");
}

void test_historical_positive(const std::vector<double>& returns) {
    assert(var::var_historical(returns) > 0);
    pass("historical_var_positive");
}

void test_historical_monotonic_confidence(const std::vector<double>& returns) {
    double v90 = var::var_historical(returns, 0.90);
    double v95 = var::var_historical(returns, 0.95);
    double v99 = var::var_historical(returns, 0.99);
    assert(v90 < v95 && v95 < v99);
    pass("historical_var_monotonic_confidence");
}

void test_historical_monotonic_horizon(const std::vector<double>& returns) {
    double v1 = var::var_historical(returns, 0.95, 1);
    double v5 = var::var_historical(returns, 0.95, 5);
    double v10 = var::var_historical(returns, 0.95, 10);
    assert(v1 < v5 && v5 < v10);
    pass("historical_var_monotonic_horizon");
}

void test_historical_cvar_geq_var(const std::vector<double>& returns) {
    for (double conf : {0.90, 0.95, 0.99}) {
        double v = var::var_historical(returns, conf);
        double cv = var::cvar_historical(returns, conf);
        assert(cv >= v);
    }
    pass("historical_cvar_geq_var");
}

void test_parametric_positive(const std::vector<double>& returns) {
    assert(var::var_parametric(returns) > 0);
    pass("parametric_var_positive");
}

void test_parametric_monotonic_confidence(const std::vector<double>& returns) {
    double v90 = var::var_parametric(returns, 0.90);
    double v95 = var::var_parametric(returns, 0.95);
    double v99 = var::var_parametric(returns, 0.99);
    assert(v90 < v95 && v95 < v99);
    pass("parametric_var_monotonic_confidence");
}

void test_parametric_cvar_geq_var(const std::vector<double>& returns) {
    for (double conf : {0.90, 0.95, 0.99}) {
        double v = var::var_parametric(returns, conf);
        double cv = var::cvar_parametric(returns, conf);
        assert(cv >= v);
    }
    pass("parametric_cvar_geq_var");
}

void test_monte_carlo_positive(const std::vector<double>& returns) {
    auto result = var::var_monte_carlo(returns);
    assert(result.var > 0);
    pass("monte_carlo_var_positive");
}

void test_monte_carlo_cvar_geq_var(const std::vector<double>& returns) {
    auto result = var::var_monte_carlo(returns, 0.95);
    assert(result.cvar >= result.var);
    pass("monte_carlo_cvar_geq_var");
}

void test_monte_carlo_seed_reproducibility(const std::vector<double>& returns) {
    auto r1 = var::var_monte_carlo(returns, 0.95, 1, 10000, 123);
    auto r2 = var::var_monte_carlo(returns, 0.95, 1, 10000, 123);
    assert(r1.var == r2.var);
    assert(r1.cvar == r2.cvar);
    pass("monte_carlo_seed_reproducibility");
}

void test_monte_carlo_simulated_returns_length(const std::vector<double>& returns) {
    auto result = var::var_monte_carlo(returns, 0.95, 1, 5000);
    assert(result.simulated_returns.size() == 5000);
    pass("monte_carlo_simulated_returns_length");
}

// ---------------------------------------------------------------------------
// Fixture validation
// ---------------------------------------------------------------------------

void test_historical_fixture(const std::vector<double>& returns) {
    auto expected = load_expected(fixtures_dir() + "/expected_historical.csv");
    for (const auto& row : expected) {
        double v = var::var_historical(returns, row.confidence, row.holding_period);
        double cv = var::cvar_historical(returns, row.confidence, row.holding_period);
        std::string label = "hist conf=" + std::to_string(row.confidence)
                          + " hp=" + std::to_string(row.holding_period);
        assert_approx(v, row.var, 0.01, label + " var");
        assert_approx(cv, row.cvar, 0.01, label + " cvar");
    }
    pass("historical_fixture_validation");
}

void test_parametric_fixture(const std::vector<double>& returns) {
    auto expected = load_expected(fixtures_dir() + "/expected_parametric.csv");
    for (const auto& row : expected) {
        double v = var::var_parametric(returns, row.confidence, row.holding_period);
        double cv = var::cvar_parametric(returns, row.confidence, row.holding_period);
        std::string label = "param conf=" + std::to_string(row.confidence)
                          + " hp=" + std::to_string(row.holding_period);
        assert_approx(v, row.var, 0.001, label + " var");
        assert_approx(cv, row.cvar, 0.001, label + " cvar");
    }
    pass("parametric_fixture_validation");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "Loading fixture data..." << std::endl;
    auto returns = load_returns(fixtures_dir() + "/sample_returns.csv");
    std::cout << "Loaded " << returns.size() << " returns\n" << std::endl;

    test_log_returns();

    test_historical_positive(returns);
    test_historical_monotonic_confidence(returns);
    test_historical_monotonic_horizon(returns);
    test_historical_cvar_geq_var(returns);

    test_parametric_positive(returns);
    test_parametric_monotonic_confidence(returns);
    test_parametric_cvar_geq_var(returns);

    test_monte_carlo_positive(returns);
    test_monte_carlo_cvar_geq_var(returns);
    test_monte_carlo_seed_reproducibility(returns);
    test_monte_carlo_simulated_returns_length(returns);

    test_historical_fixture(returns);
    test_parametric_fixture(returns);

    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed" << std::endl;
    return (tests_passed == tests_run) ? 0 : 1;
}
