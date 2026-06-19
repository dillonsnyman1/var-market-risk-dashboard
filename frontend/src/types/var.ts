export type VarMethod = "historical" | "parametric" | "monte_carlo";

export interface VarResult {
  confidence: number;
  holding_period: number;
  var_historical: number;
  cvar_historical: number;
  var_parametric: number;
  cvar_parametric: number;
  var_monte_carlo: number;
  cvar_monte_carlo: number;
}

export interface ReturnStats {
  count: number;
  mean_daily: number;
  std_daily: number;
  mean_annual: number;
  std_annual: number;
  skewness: number;
  kurtosis: number;
  min_return: number;
  max_return: number;
}

export interface ReturnDistribution {
  bin_edges: number[];
  counts: number[];
  normal_pdf: number[];
}

export interface VarResponse {
  stats: ReturnStats;
  var_surface: VarResult[];
  distribution: ReturnDistribution;
}

export interface TickerResponse {
  ticker: string;
  returns: number[];
  dates: string[];
  prices: number[];
}

export interface BacktestResponse {
  dates: number[];
  actual_returns: number[];
  var_predictions: number[];
  breaches: boolean[];
  breach_count: number;
  breach_rate: number;
  expected_breach_rate: number;
  kupiec_p_value: number;
}

export interface MonteCarloPathsResponse {
  time_points: number[];
  paths: number[][];
  var_95: number;
  var_99: number;
  terminal_returns: number[];
}

export const METHOD_LABELS: Record<VarMethod, string> = {
  historical: "Historical Simulation",
  parametric: "Variance-Covariance",
  monte_carlo: "Monte Carlo",
};

export const METHOD_COLORS: Record<VarMethod, string> = {
  historical: "#2563eb",
  parametric: "#7c3aed",
  monte_carlo: "#0891b2",
};

export const DEFAULT_TICKER = "AAPL";
export const DEFAULT_PERIOD = "2y";
