import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReturnDistribution, VarResult } from "../types/var";

interface Props {
  distribution: ReturnDistribution;
  surface: VarResult[];
}

export function ReturnDistributionChart({ distribution, surface }: Props) {
  const { bin_edges, counts, normal_pdf } = distribution;
  const data = counts.map((count, i) => {
    const center = (bin_edges[i] + bin_edges[i + 1]) / 2;
    return {
      center: +(center * 100).toFixed(3),
      count,
      normal: +normal_pdf[i].toFixed(2),
    };
  });

  const varLines = surface
    .filter((r) => r.holding_period === 1)
    .map((r) => ({
      confidence: r.confidence,
      value: -(r.var_historical * 100),
    }));

  return (
    <div className="chart-card wide">
      <h3>Return Distribution</h3>
      <p className="chart-subtitle">
        Daily log return histogram with fitted normal density. Vertical lines show
        the Historical VaR threshold at each confidence level - returns beyond
        these lines are in the loss tail.
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ left: 10, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="center"
            tick={{ fontSize: 11 }}
            label={{ value: "Return (%)", position: "insideBottom", offset: -10, fontSize: 12 }}
          />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip
            formatter={(v, name) =>
              name === "normal" ? [Number(v).toFixed(1), "Normal fit"] : [v, "Count"]
            }
            labelFormatter={(l) => `${l}%`}
          />
          <Bar dataKey="count" fill="#2563eb" opacity={0.6} />
          <Line dataKey="normal" stroke="#b91c1c" dot={false} strokeWidth={2} name="Normal fit" />
          {varLines.map((vl) => (
            <Line
              key={vl.confidence}
              data={[
                { center: +vl.value.toFixed(3), count: 0, normal: 0 },
              ]}
              dataKey="count"
              stroke="transparent"
              dot={false}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
