function [v, cv] = var_historical(returns, confidence, holding_period)
% VAR_HISTORICAL  Empirical percentile VaR and CVaR.
%   [V, CV] = VAR_HISTORICAL(RETURNS, CONFIDENCE, HOLDING_PERIOD)
%
%   Multi-day scaled by sqrt(t).

    if nargin < 2, confidence = 0.95; end
    if nargin < 3, holding_period = 1; end

    sorted = sort(returns);
    n = length(sorted);
    idx = (1 - confidence) * (n - 1) + 1;
    lo = floor(idx);
    hi = ceil(idx);
    if lo == hi
        threshold = sorted(lo);
    else
        frac = idx - lo;
        threshold = sorted(lo) * (1 - frac) + sorted(hi) * frac;
    end

    var_1d = -threshold;
    v = var_1d * sqrt(holding_period);

    tail = sorted(sorted <= threshold);
    if ~isempty(tail)
        cvar_1d = -mean(tail);
    else
        cvar_1d = -threshold;
    end
    cv = cvar_1d * sqrt(holding_period);
end
