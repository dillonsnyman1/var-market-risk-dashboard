import { useEffect, useState } from "react";
import "./App.css";
import { computeVar, fetchTicker, uploadCsv } from "./api/client";
import { BacktestChart } from "./components/BacktestChart";
import { DataSourcePanel } from "./components/DataSourcePanel";
import { MonteCarloPathsChart } from "./components/MonteCarloPathsChart";
import { ReturnDistributionChart } from "./components/ReturnDistributionChart";
import { VarComparisonChart } from "./components/VarComparisonChart";
import { VarSummaryCards } from "./components/VarSummaryCards";
import { VarSurfaceTable } from "./components/VarSurfaceTable";
import type { VarResponse } from "./types/var";
import { DEFAULT_PERIOD, DEFAULT_TICKER } from "./types/var";

type Tab = "overview" | "distribution" | "monte-carlo" | "backtest";

const TABS: [Tab, string][] = [
  ["overview", "VaR Overview"],
  ["distribution", "Return Distribution"],
  ["monte-carlo", "Monte Carlo Simulation"],
  ["backtest", "Backtesting"],
];

function App() {
  const [returns, setReturns] = useState<number[] | null>(null);
  const [varData, setVarData] = useState<VarResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("overview");

  function loadTicker(ticker: string, period: string) {
    setLoading(true);
    setError(null);
    fetchTicker(ticker, period)
      .then((res) => {
        setReturns(res.returns);
        setInfo(`${res.ticker} - ${res.returns.length} daily returns loaded`);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to fetch ticker"))
      .finally(() => setLoading(false));
  }

  function handleUpload(file: File) {
    setLoading(true);
    setError(null);
    uploadCsv(file)
      .then((res) => {
        setReturns(res.returns);
        setInfo(`CSV uploaded - ${res.returns.length} returns loaded`);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Upload failed"))
      .finally(() => setLoading(false));
  }

  // load default ticker on mount
  useEffect(() => {
    loadTicker(DEFAULT_TICKER, DEFAULT_PERIOD);
  }, []);

  // recompute VaR surface whenever the return series changes
  useEffect(() => {
    if (!returns || returns.length < 30) return;
    setLoading(true);
    computeVar(returns)
      .then(setVarData)
      .catch((e) => setError(e instanceof Error ? e.message : "VaR computation failed"))
      .finally(() => setLoading(false));
  }, [returns]);

  return (
    <>
      <header className="app-header">
        <h1>VaR & Market Risk Dashboard</h1>
        <p className="header-tagline">
          Single-asset Value at Risk and Expected Shortfall using three methods side-by-side.
        </p>
        <p className="header-background">
          Enter a ticker symbol to fetch historical data from Yahoo Finance, or
          upload a CSV with a price or return column. The dashboard computes VaR
          and CVaR using Historical Simulation, Variance-Covariance (Parametric),
          and Monte Carlo methods across multiple confidence levels and holding
          periods.
        </p>
      </header>

      <DataSourcePanel
        onTickerLoad={loadTicker}
        onFileUpload={handleUpload}
        loading={loading}
        info={info}
      />

      {error && <div className="status-message error">{error}</div>}

      {loading && !varData && <div className="status-message">Loading...</div>}

      {varData && returns && (
        <>
          <VarSummaryCards surface={varData.var_surface} />

          <nav className="tab-nav">
            {TABS.map(([key, label]) => (
              <button
                key={key}
                className={`tab-button ${tab === key ? "active" : ""}`}
                onClick={() => setTab(key)}
              >
                {label}
              </button>
            ))}
          </nav>

          <div className="tab-content">
            {tab === "overview" && (
              <>
                <div className="charts-row">
                  <VarComparisonChart surface={varData.var_surface} />
                </div>
                <VarSurfaceTable surface={varData.var_surface} />
              </>
            )}

            {tab === "distribution" && (
              <ReturnDistributionChart
                distribution={varData.distribution}
                surface={varData.var_surface}
              />
            )}

            {tab === "monte-carlo" && <MonteCarloPathsChart returns={returns} />}

            {tab === "backtest" && <BacktestChart returns={returns} />}
          </div>
        </>
      )}
    </>
  );
}

export default App;
