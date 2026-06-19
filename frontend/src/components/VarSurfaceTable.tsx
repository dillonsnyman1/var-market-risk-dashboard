import type { VarResult } from "../types/var";

interface Props {
  surface: VarResult[];
}

const pct = (v: number) => (v * 100).toFixed(2) + "%";

export function VarSurfaceTable({ surface }: Props) {
  return (
    <div className="chart-card wide">
      <h3>VaR Surface</h3>
      <p className="chart-subtitle">
        VaR and CVaR across all methods, confidence levels, and holding periods.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="var-surface-table">
          <thead>
            <tr>
              <th>Confidence</th>
              <th>Horizon</th>
              <th>Hist VaR</th>
              <th>Hist CVaR</th>
              <th>Param VaR</th>
              <th>Param CVaR</th>
              <th>MC VaR</th>
              <th>MC CVaR</th>
            </tr>
          </thead>
          <tbody>
            {surface.map((r, i) => (
              <tr key={i}>
                <td className="row-header">{(r.confidence * 100).toFixed(0)}%</td>
                <td className="row-header">{r.holding_period}-day</td>
                <td>{pct(r.var_historical)}</td>
                <td>{pct(r.cvar_historical)}</td>
                <td>{pct(r.var_parametric)}</td>
                <td>{pct(r.cvar_parametric)}</td>
                <td>{pct(r.var_monte_carlo)}</td>
                <td>{pct(r.cvar_monte_carlo)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
