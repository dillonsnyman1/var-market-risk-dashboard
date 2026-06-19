import type { VarResult } from "../types/var";

interface Props {
  surface: VarResult[];
}

const pct = (v: number) => (v * 100).toFixed(2) + "%";

export function VarSummaryCards({ surface }: Props) {
  const row95 = surface.find((r) => r.confidence === 0.95 && r.holding_period === 1);
  if (!row95) return null;

  const cards = [
    {
      label: "Historical VaR (95%, 1-day)",
      value: pct(row95.var_historical),
      sub: `CVaR: ${pct(row95.cvar_historical)}`,
      color: "#2563eb",
    },
    {
      label: "Parametric VaR (95%, 1-day)",
      value: pct(row95.var_parametric),
      sub: `CVaR: ${pct(row95.cvar_parametric)}`,
      color: "#7c3aed",
    },
    {
      label: "Monte Carlo VaR (95%, 1-day)",
      value: pct(row95.var_monte_carlo),
      sub: `CVaR: ${pct(row95.cvar_monte_carlo)}`,
      color: "#0891b2",
    },
    {
      label: "Expected Shortfall (95%, 1-day)",
      value: pct(
        (row95.cvar_historical + row95.cvar_parametric + row95.cvar_monte_carlo) / 3
      ),
      sub: "Average CVaR across methods",
      color: "#b91c1c",
    },
  ];

  return (
    <div className="summary-cards">
      {cards.map((c) => (
        <div key={c.label} className="summary-card" style={{ borderTopColor: c.color }}>
          <div className="summary-card-label">{c.label}</div>
          <div className="summary-card-value">{c.value}</div>
          <div className="summary-card-subvalue">{c.sub}</div>
        </div>
      ))}
    </div>
  );
}
