import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { VarResult } from "../types/var";
import { METHOD_COLORS } from "../types/var";

interface Props {
  surface: VarResult[];
}

export function VarComparisonChart({ surface }: Props) {
  const data = surface
    .filter((r) => r.holding_period === 1)
    .map((r) => ({
      name: `${(r.confidence * 100).toFixed(0)}%`,
      Historical: +(r.var_historical * 100).toFixed(3),
      Parametric: +(r.var_parametric * 100).toFixed(3),
      "Monte Carlo": +(r.var_monte_carlo * 100).toFixed(3),
    }));

  return (
    <div className="chart-card">
      <h3>VaR by Method (1-day)</h3>
      <p className="chart-subtitle">
        Comparing the three methods at each confidence level.
      </p>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} unit="%" />
          <Tooltip formatter={(v) => Number(v).toFixed(3) + "%"} />
          <Legend />
          <Bar dataKey="Historical" fill={METHOD_COLORS.historical} radius={[4, 4, 0, 0]} />
          <Bar dataKey="Parametric" fill={METHOD_COLORS.parametric} radius={[4, 4, 0, 0]} />
          <Bar dataKey="Monte Carlo" fill={METHOD_COLORS.monte_carlo} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
