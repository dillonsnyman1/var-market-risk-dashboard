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
| `GET`  | `/api/health` | Health check |

## Data flow

The ticker and upload endpoints return raw log returns to the frontend.
The frontend then sends those returns to the computation endpoints as
needed. Nothing is stored server-side between requests.

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
