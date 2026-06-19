import { useEffect, useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { runBacktest } from "../api/client";
import type { BacktestResponse } from "../types/var";

interface Props {
  returns: number[];
}

export function BacktestChart({ returns }: Props) {
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (returns.length < 260) {
      setError("Need at least 260 returns for backtesting (250-day window)");
      return;
    }
    setLoading(true);
    setError(null);
    runBacktest(returns, 0.95, 250, "historical")
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed"))
      .finally(() => setLoading(false));
  }, [returns]);

  if (loading) return <div className="chart-card"><div className="status-message">Running backtest...</div></div>;
  if (error) return <div className="chart-card"><div className="status-message error">{error}</div></div>;
  if (!data) return null;

  // convert to percentages for display; breach is null on non-breach days
  // so recharts only renders scatter dots where there's a value
  const chartData = data.dates.map((d, i) => ({
    index: d,
    actual: +(data.actual_returns[i] * 100).toFixed(4),
    var: +(data.var_predictions[i] * 100).toFixed(4),
    breach: data.breaches[i] ? +(data.actual_returns[i] * 100).toFixed(4) : null,
  }));

  const kupiecPass = data.kupiec_p_value > 0.05;

  return (
    <div className="chart-card wide">
      <h3>Backtesting</h3>
      <p className="chart-subtitle">
        Rolling 250-day Historical VaR at 95% confidence vs actual returns.
        Red dots mark days where the actual loss exceeded the predicted VaR.
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={chartData} margin={{ left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="index" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} unit="%" />
          <Tooltip
            formatter={(v, name) => {
              const val = Number(v).toFixed(3) + "%";
              if (name === "breach") return [val, "Breach"];
              return [val, name === "var" ? "VaR threshold" : "Actual return"];
            }}
          />
          <Line dataKey="actual" stroke="#94a3b8" strokeWidth={0.8} dot={false} name="Actual return" />
          <Line dataKey="var" stroke="#2563eb" strokeWidth={1.5} dot={false} name="VaR threshold" />
          <Scatter dataKey="breach" fill="#b91c1c" name="Breach" />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="backtest-stats">
        <div>
          <div className="backtest-stat-label">Breaches</div>
          <div className="backtest-stat-value">{data.breach_count} / {data.dates.length}</div>
        </div>
        <div>
          <div className="backtest-stat-label">Breach Rate</div>
          <div className="backtest-stat-value">{(data.breach_rate * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="backtest-stat-label">Expected Rate</div>
          <div className="backtest-stat-value">{(data.expected_breach_rate * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="backtest-stat-label">Kupiec p-value</div>
          <div className="backtest-stat-value">{data.kupiec_p_value.toFixed(4)}</div>
        </div>
        <div>
          <div className="backtest-stat-label">Kupiec Test</div>
          <div className="backtest-stat-value" style={{ color: kupiecPass ? "#16a34a" : "#b91c1c" }}>
            {kupiecPass ? "PASS" : "FAIL"}
          </div>
        </div>
      </div>
    </div>
  );
}
