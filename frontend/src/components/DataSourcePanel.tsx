import { useState } from "react";
import { DEFAULT_TICKER, DEFAULT_PERIOD } from "../types/var";

interface Props {
  onTickerLoad: (ticker: string, period: string) => void;
  onFileUpload: (file: File) => void;
  loading: boolean;
  info: string | null;
}

export function DataSourcePanel({ onTickerLoad, onFileUpload, loading, info }: Props) {
  const [ticker, setTicker] = useState(DEFAULT_TICKER);
  const [period, setPeriod] = useState(DEFAULT_PERIOD);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (ticker.trim()) onTickerLoad(ticker.trim(), period);
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onFileUpload(file);
    e.target.value = "";
  }

  return (
    <div className="toolbar">
      <form className="data-source-form" onSubmit={handleSubmit}>
        <label className="form-field">
          Ticker
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="AAPL"
          />
        </label>
        <label className="form-field">
          Period
          <select value={period} onChange={(e) => setPeriod(e.target.value)}>
            <option value="1y">1 year</option>
            <option value="2y">2 years</option>
            <option value="5y">5 years</option>
          </select>
        </label>
        <button type="submit" className="apply-button" disabled={loading}>
          {loading ? "Loading..." : "Fetch Data"}
        </button>
        <label className="upload-label">
          Upload CSV
          <input type="file" accept=".csv" onChange={handleFile} hidden />
        </label>
      </form>
      {info && <div className="data-source-info">{info}</div>}
    </div>
  );
}
