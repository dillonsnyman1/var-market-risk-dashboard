import type {
  BacktestResponse,
  MonteCarloPathsResponse,
  TickerResponse,
  VarResponse,
} from "../types/var";

// set VITE_API_BASE_URL at build time for prod; falls back to local dev server
const API_BASE: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    throw new Error(
      (payload as { detail?: string } | null)?.detail ??
        `Request failed (${res.status})`
    );
  }
  return res.json() as Promise<T>;
}

export function fetchTicker(
  ticker: string,
  period: string
): Promise<TickerResponse> {
  return post<TickerResponse>("/api/ticker", { ticker, period });
}

export async function uploadCsv(file: File): Promise<TickerResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    throw new Error(
      (payload as { detail?: string } | null)?.detail ??
        `Upload failed (${res.status})`
    );
  }
  return res.json() as Promise<TickerResponse>;
}

export function computeVar(
  returns: number[],
  confidences: number[] = [0.9, 0.95, 0.99],
  holdingPeriods: number[] = [1, 5, 10],
  nSimulations: number = 10_000
): Promise<VarResponse> {
  return post<VarResponse>("/api/var", {
    returns,
    confidences,
    holding_periods: holdingPeriods,
    n_simulations: nSimulations,
  });
}

export function runBacktest(
  returns: number[],
  confidence: number = 0.95,
  window: number = 250,
  method: string = "historical"
): Promise<BacktestResponse> {
  return post<BacktestResponse>("/api/backtest", {
    returns,
    confidence,
    window,
    method,
  });
}

export function fetchMonteCarloPaths(
  returns: number[],
  holdingPeriod: number = 10,
  nSimulations: number = 500
): Promise<MonteCarloPathsResponse> {
  return post<MonteCarloPathsResponse>("/api/monte-carlo-paths", {
    returns,
    holding_period: holdingPeriod,
    n_simulations: nSimulations,
  });
}
