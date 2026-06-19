# Frontend

React + Vite + TypeScript. No UI framework - vanilla CSS with custom
properties matching the other projects in this portfolio.

## Tabs

- **VaR Overview** - grouped bar chart comparing the three methods at
  each confidence level, plus the full VaR surface table.
- **Return Distribution** - histogram of daily log returns with a
  fitted normal density overlay.
- **Monte Carlo Simulation** - 50 simulated GBM price paths over a
  10-day horizon.
- **Backtesting** - rolling 250-day VaR predictions vs actual returns,
  with breach markers and Kupiec test results.

## Running locally

```bash
npm install
npm run dev
```

Expects the backend running on http://localhost:8000. Set
`VITE_API_BASE_URL` to override (e.g. for a deployed backend).

## Build

```bash
npm run build
```

Output goes to `dist/`, which gets synced to S3 + CloudFront in Phase 3.

## Charts

All charts use [Recharts](https://recharts.org/). The TypeScript types
in `src/types/var.ts` mirror the backend Pydantic models.
