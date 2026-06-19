# Extensions

Planned enhancements and future development directions. The current
implementation covers single-asset VaR - the items below extend it to
portfolio-level risk and richer dynamics.

---

## Multi-asset portfolio VaR

The most natural next step. Extends the three existing methods to a
portfolio of N assets with weights w₁, ..., wₙ.

**Historical Simulation**: compute the portfolio return series as the
weighted sum of individual asset returns, then apply the existing
percentile-based VaR/CVaR calculation unchanged.

**Variance-Covariance (Parametric)**: estimate the N×N covariance matrix Σ
from historical returns. Portfolio variance is σ²_p = wᵀΣw, and VaR follows
from the portfolio sigma.

**Monte Carlo**: simulate correlated returns using the Cholesky
decomposition of Σ. Each simulated vector of N asset returns preserves the
pairwise correlation structure. Portfolio return is the weighted sum, then
VaR/CVaR are computed as before.

### Decomposition

- **Component VaR**: wᵢ × ∂VaR/∂wᵢ - each asset's contribution to total
  VaR. Component VaRs sum to total portfolio VaR.
- **Marginal VaR**: ∂VaR/∂wᵢ - sensitivity of portfolio VaR to a small
  increase in weight i.
- **Incremental VaR**: VaR(portfolio with asset i) − VaR(portfolio without
  asset i) - the discrete impact of adding or removing an asset.
- **Diversification benefit**: Σᵢ VaRᵢ − VaR_portfolio - the risk reduction
  from correlation being less than 1.

### Visualisations

- Correlation heatmap (N×N matrix)
- Component VaR waterfall chart
- Diversification benefit bar chart
- Efficient frontier with VaR on the x-axis

---

## Stress testing

Predefined historical and hypothetical scenarios that apply a vector of
factor shocks to the portfolio:

- **GFC 2008**: equity -40%, vol +150%, credit spreads +300bps
- **COVID 2020**: equity -35%, vol +200%, rates -100bps
- **Rate shock**: parallel yield curve shift +200bps
- **Vol spike**: implied vol doubles, spot unchanged

Display stressed VaR vs. normal VaR for each scenario.

---

## Conditional volatility (GARCH)

Replace the constant-σ assumption with time-varying volatility:

**GARCH(1,1)**: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Estimate parameters via maximum likelihood, then use the current
conditional volatility for forward-looking VaR. This produces VaR
estimates that react faster to recent market stress.

**EWMA** (Exponentially Weighted Moving Average): a simplified special case
of GARCH where ω = 0 and α + β = 1, controlled by a single decay factor λ
(RiskMetrics uses λ = 0.94 for daily data).

---

## Additional reference languages

Julia and Java would be useful additions to the `reference/` folder:

- **Julia**: increasingly common in quant research - fast, expressive, good
  numerical libraries.
- **Java/Kotlin**: still the dominant language in bank risk systems and
  trade capture platforms.
