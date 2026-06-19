from enum import Enum

from pydantic import BaseModel, Field


class VarMethod(str, Enum):
    historical = "historical"
    parametric = "parametric"
    monte_carlo = "monte_carlo"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TickerRequest(BaseModel):
    ticker: str = Field(description="Yahoo Finance ticker symbol, e.g. AAPL")
    period: str = Field(default="2y", description="History period, e.g. 1y, 2y, 5y")


class VarRequest(BaseModel):
    returns: list[float] = Field(description="Array of log returns")
    confidences: list[float] = Field(default=[0.90, 0.95, 0.99])
    holding_periods: list[int] = Field(default=[1, 5, 10])
    n_simulations: int = Field(default=10_000, ge=1_000, le=100_000)


class BacktestRequest(BaseModel):
    returns: list[float]
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    window: int = Field(default=250, ge=50, le=1000)
    method: VarMethod = Field(default=VarMethod.historical)


class MonteCarloPathsRequest(BaseModel):
    returns: list[float]
    holding_period: int = Field(default=10, ge=1, le=252)
    n_simulations: int = Field(default=500, ge=100, le=5_000)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class VarResult(BaseModel):
    confidence: float
    holding_period: int
    var_historical: float
    cvar_historical: float
    var_parametric: float
    cvar_parametric: float
    var_monte_carlo: float
    cvar_monte_carlo: float


class ReturnStats(BaseModel):
    count: int
    mean_daily: float
    std_daily: float
    mean_annual: float
    std_annual: float
    skewness: float
    kurtosis: float
    min_return: float
    max_return: float


class ReturnDistribution(BaseModel):
    bin_edges: list[float]
    counts: list[int]
    normal_pdf: list[float]


class VarResponse(BaseModel):
    stats: ReturnStats
    var_surface: list[VarResult]
    distribution: ReturnDistribution


class TickerResponse(BaseModel):
    ticker: str
    returns: list[float]
    dates: list[str]
    prices: list[float]


class BacktestResponse(BaseModel):
    dates: list[int]
    actual_returns: list[float]
    var_predictions: list[float]
    breaches: list[bool]
    breach_count: int
    breach_rate: float
    expected_breach_rate: float
    kupiec_p_value: float


class MonteCarloPathsResponse(BaseModel):
    time_points: list[float]
    paths: list[list[float]]
    var_95: float
    var_99: float
    terminal_returns: list[float]
