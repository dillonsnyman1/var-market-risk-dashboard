#!/usr/bin/env Rscript
# Tests for the R VaR implementation.
# Run with: Rscript test_var.R

library(testthat)

# Source var.R relative to this script's location.
this_dir <- tryCatch(
  dirname(normalizePath(sys.frame(0)$ofile)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
      dirname(normalizePath(sub("^--file=", "", file_arg)))
    } else {
      getwd()
    }
  }
)

source(file.path(this_dir, "var.R"))

FIXTURES_DIR <- normalizePath(file.path(this_dir, "..", "fixtures"))

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

load_returns <- function() {
  df <- read.csv(file.path(FIXTURES_DIR, "sample_returns.csv"))
  df$return
}

load_expected <- function(filename) {
  read.csv(file.path(FIXTURES_DIR, filename))
}

# ---------------------------------------------------------------------------
# log_returns
# ---------------------------------------------------------------------------

test_that("log_returns has correct length", {
  prices <- c(100, 105, 103, 107)
  lr <- log_returns(prices)
  expect_equal(length(lr), length(prices) - 1)
})

test_that("log_returns has correct values", {
  prices <- c(100, 110)
  lr <- log_returns(prices)
  expect_equal(lr[1], log(1.1))
})

# ---------------------------------------------------------------------------
# Historical Simulation
# ---------------------------------------------------------------------------

test_that("historical VaR is positive", {
  r <- load_returns()
  expect_gt(var_historical(r), 0)
})

test_that("historical VaR increases with confidence", {
  r <- load_returns()
  v90 <- var_historical(r, 0.90)
  v95 <- var_historical(r, 0.95)
  v99 <- var_historical(r, 0.99)
  expect_lt(v90, v95)
  expect_lt(v95, v99)
})

test_that("historical VaR increases with holding period", {
  r <- load_returns()
  v1  <- var_historical(r, holding_period = 1)
  v5  <- var_historical(r, holding_period = 5)
  v10 <- var_historical(r, holding_period = 10)
  expect_lt(v1, v5)
  expect_lt(v5, v10)
})

test_that("historical CVaR >= VaR", {
  r <- load_returns()
  for (conf in c(0.90, 0.95, 0.99)) {
    v  <- var_historical(r, conf)
    cv <- cvar_historical(r, conf)
    expect_gte(cv, v)
  }
})

# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------

test_that("parametric VaR is positive", {
  r <- load_returns()
  expect_gt(var_parametric(r), 0)
})

test_that("parametric VaR increases with confidence", {
  r <- load_returns()
  v90 <- var_parametric(r, 0.90)
  v95 <- var_parametric(r, 0.95)
  v99 <- var_parametric(r, 0.99)
  expect_lt(v90, v95)
  expect_lt(v95, v99)
})

test_that("parametric CVaR >= VaR", {
  r <- load_returns()
  for (conf in c(0.90, 0.95, 0.99)) {
    v  <- var_parametric(r, conf)
    cv <- cvar_parametric(r, conf)
    expect_gte(cv, v)
  }
})

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

test_that("MC VaR is positive", {
  r <- load_returns()
  mc <- var_monte_carlo(r)
  expect_gt(mc$var, 0)
})

test_that("MC CVaR >= VaR", {
  r <- load_returns()
  mc <- var_monte_carlo(r, 0.95)
  expect_gte(mc$cvar, mc$var)
})

test_that("MC is seed-reproducible", {
  r <- load_returns()
  mc1 <- var_monte_carlo(r, seed = 123)
  mc2 <- var_monte_carlo(r, seed = 123)
  expect_equal(mc1$var, mc2$var)
  expect_equal(mc1$cvar, mc2$cvar)
})

test_that("MC returns correct number of simulations", {
  r <- load_returns()
  mc <- var_monte_carlo(r, n_simulations = 5000)
  expect_equal(length(mc$simulated_returns), 5000)
})

# ---------------------------------------------------------------------------
# Fixture validation - historical
# ---------------------------------------------------------------------------

test_that("historical matches fixture values", {
  r <- load_returns()
  expected <- load_expected("expected_historical.csv")
  for (i in seq_len(nrow(expected))) {
    row <- expected[i, ]
    v  <- var_historical(r, row$confidence, row$holding_period)
    cv <- cvar_historical(r, row$confidence, row$holding_period)
    expect_equal(v, row$var, tolerance = 0.01,
                 label = sprintf("hist VaR conf=%.2f hp=%d", row$confidence, row$holding_period))
    expect_equal(cv, row$cvar, tolerance = 0.01,
                 label = sprintf("hist CVaR conf=%.2f hp=%d", row$confidence, row$holding_period))
  }
})

# ---------------------------------------------------------------------------
# Fixture validation - parametric
# ---------------------------------------------------------------------------

test_that("parametric matches fixture values", {
  r <- load_returns()
  expected <- load_expected("expected_parametric.csv")
  for (i in seq_len(nrow(expected))) {
    row <- expected[i, ]
    v  <- var_parametric(r, row$confidence, row$holding_period)
    cv <- cvar_parametric(r, row$confidence, row$holding_period)
    expect_equal(v, row$var, tolerance = 0.001,
                 label = sprintf("param VaR conf=%.2f hp=%d", row$confidence, row$holding_period))
    expect_equal(cv, row$cvar, tolerance = 0.001,
                 label = sprintf("param CVaR conf=%.2f hp=%d", row$confidence, row$holding_period))
  }
})
