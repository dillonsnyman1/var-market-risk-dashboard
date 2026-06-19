import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchMonteCarloPaths } from "../api/client";
import type { MonteCarloPathsResponse } from "../types/var";

interface Props {
  returns: number[];
}

export function MonteCarloPathsChart({ returns }: Props) {
  const [data, setData] = useState<MonteCarloPathsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchMonteCarloPaths(returns, 10, 200)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed"))
      .finally(() => setLoading(false));
  }, [returns]);

  if (loading) return <div className="chart-card"><div className="status-message">Simulating paths...</div></div>;
  if (error) return <div className="chart-card"><div className="status-message error">{error}</div></div>;
  if (!data) return null;

  // cap at 50 paths or the chart becomes unreadable
  const pathsToShow = Math.min(50, data.paths.length);

  // recharts needs one key per line, so we pivot paths into columns
  const chartData = data.time_points.map((t, ti) => {
    const point: Record<string, number> = { day: t };
    for (let p = 0; p < pathsToShow; p++) {
      point[`p${p}`] = data.paths[p][ti];
    }
    return point;
  });

  return (
    <div className="chart-card wide">
      <h3>Monte Carlo Simulation</h3>
      <p className="chart-subtitle">
        {pathsToShow} simulated GBM price paths over a 10-day horizon (normalised,
        starting at 1.0). VaR(95%) = {(data.var_95 * 100).toFixed(2)}%,
        VaR(99%) = {(data.var_99 * 100).toFixed(2)}%.
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={chartData} margin={{ left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="day"
            tick={{ fontSize: 12 }}
            label={{ value: "Day", position: "insideBottom", offset: -5, fontSize: 12 }}
          />
          <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} />
          <Tooltip labelFormatter={(l) => `Day ${l}`} />
          {Array.from({ length: pathsToShow }, (_, i) => (
            <Line
              key={i}
              dataKey={`p${i}`}
              stroke="#2563eb"
              strokeWidth={0.5}
              strokeOpacity={0.15}
              dot={false}
              name={`Path ${i + 1}`}
              legendType="none"
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
