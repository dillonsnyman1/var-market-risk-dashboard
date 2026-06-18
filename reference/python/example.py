"""
Usage demo for the VaR reference implementation.

Loads the shared fixture data and prints VaR/CVaR across all methods,
confidence levels, and holding periods, followed by a backtest summary.

Run:
    python example.py
"""

from __future__ import annotations

import os

import pandas as pd

from var import compute_var_surface, backtest_var

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def main() -> None:
    returns = pd.read_csv(os.path.join(FIXTURES, "sample_returns.csv"))["return"].values
    print(f"Loaded {len(returns)} daily log returns from fixture data\n")

    # ------------------------------------------------------------------
    # VaR surface
    # ------------------------------------------------------------------
    surface = compute_var_surface(returns)

    print("=" * 90)
    print("VaR & CVaR Surface (all values expressed as positive loss percentages)")
    print("=" * 90)

    for conf in [0.90, 0.95, 0.99]:
        rows = surface[surface["confidence"] == conf]
        print(f"\n  Confidence: {conf:.0%}")
        print(f"  {'Holding':>10s}  {'--- Historical ---':>20s}  {'--- Parametric ---':>20s}  {'--- Monte Carlo ---':>20s}")
        print(f"  {'Period':>10s}  {'VaR':>9s}  {'CVaR':>9s}  {'VaR':>9s}  {'CVaR':>9s}  {'VaR':>9s}  {'CVaR':>9s}")
        print(f"  {'':->10s}  {'':->9s}  {'':->9s}  {'':->9s}  {'':->9s}  {'':->9s}  {'':->9s}")
        for _, r in rows.iterrows():
            print(
                f"  {int(r['holding_period']):>7d}-day"
                f"  {r['var_historical']:>8.4%}"
                f"  {r['cvar_historical']:>8.4%}"
                f"  {r['var_parametric']:>8.4%}"
                f"  {r['cvar_parametric']:>8.4%}"
                f"  {r['var_monte_carlo']:>8.4%}"
                f"  {r['cvar_monte_carlo']:>8.4%}"
            )

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 90)
    print("Backtest Results (95% confidence, 250-day rolling window)")
    print("=" * 90)

    for method in ["historical", "parametric"]:
        bt = backtest_var(returns, confidence=0.95, window=250, method=method)
        status = "PASS" if bt.attrs["kupiec_p_value"] > 0.05 else "FAIL"
        print(f"\n  Method: {method.title()}")
        print(f"    Observations:         {len(bt)}")
        print(f"    Breach count:         {bt.attrs['breach_count']}")
        print(f"    Breach rate:          {bt.attrs['breach_rate']:.2%}")
        print(f"    Expected rate:        {bt.attrs['expected_breach_rate']:.2%}")
        print(f"    Kupiec p-value:       {bt.attrs['kupiec_p_value']:.4f}")
        print(f"    Kupiec test:          {status}")


if __name__ == "__main__":
    main()
