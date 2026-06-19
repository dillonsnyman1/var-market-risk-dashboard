function result = var_monte_carlo(returns, confidence, holding_period, n_simulations, seed)
% VAR_MONTE_CARLO  GBM-based Monte Carlo VaR and CVaR.
%   RESULT = VAR_MONTE_CARLO(RETURNS, CONFIDENCE, HOLDING_PERIOD, N_SIMULATIONS, SEED)
%
%   Returns a struct with fields: var, cvar, simulated_returns.

    if nargin < 2, confidence = 0.95; end
    if nargin < 3, holding_period = 1; end
    if nargin < 4, n_simulations = 10000; end
    if nargin < 5, seed = 42; end

    mu = mean(returns);
    sigma = std(returns);
    dt = holding_period;
    drift = (mu - 0.5 * sigma^2) * dt;
    diffusion = sigma * sqrt(dt);

    rng(seed);
    z = randn(n_simulations, 1);
    sim_returns = drift + diffusion * z;

    sorted = sort(sim_returns);
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

    var_val = -threshold;
    tail = sim_returns(sim_returns <= -var_val);
    if ~isempty(tail)
        cvar_val = -mean(tail);
    else
        cvar_val = var_val;
    end

    result.var = var_val;
    result.cvar = cvar_val;
    result.simulated_returns = sim_returns;
end
