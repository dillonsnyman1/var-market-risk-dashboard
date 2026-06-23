# Backend

Python + FastAPI. Fully stateless - no database, all computation done
in-request.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/ticker` | Fetch historical prices via Yahoo Finance, return log returns |
| `POST` | `/api/upload` | Parse an uploaded CSV (prices or returns) |
| `POST` | `/api/var` | Compute full VaR/CVaR surface across methods, confidences, horizons |
| `POST` | `/api/backtest` | Rolling-window backtest with Kupiec POF test |
| `POST` | `/api/monte-carlo-paths` | Simulate GBM price paths for visualisation |
| `POST` | `/api/portfolio-tickers` | Fetch multiple tickers, align on shared trading dates |
| `POST` | `/api/portfolio-var` | Portfolio VaR surface + correlation + risk decomposition |
| `GET`  | `/api/health` | Health check |

## Data flow

**Single asset:** The ticker and upload endpoints return raw log returns
to the frontend. The frontend then sends those returns to the
computation endpoints as needed.

**Portfolio:** The portfolio-tickers endpoint fetches multiple tickers,
aligns them on shared trading dates (inner join), and returns per-asset
return series. The frontend passes these to portfolio-var, which
computes the full VaR surface, correlation matrix, and component/
marginal/incremental VaR decomposition.

Nothing is stored server-side between requests.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs at http://localhost:8000/docs.

## Tests

```bash
pytest
```

## Deployment

The Dockerfile builds a Lambda container image (arm64). Mangum wraps
the FastAPI app as a Lambda handler. See `../infra/` for the Terraform
config (Phase 3).
