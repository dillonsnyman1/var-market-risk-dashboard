function [v, cv] = var_parametric(returns, confidence, holding_period)
% VAR_PARAMETRIC  Closed-form VaR and CVaR assuming N(mu, sigma^2).
%   [V, CV] = VAR_PARAMETRIC(RETURNS, CONFIDENCE, HOLDING_PERIOD)

    if nargin < 2, confidence = 0.95; end
    if nargin < 3, holding_period = 1; end

    mu = mean(returns);
    sigma = std(returns);  % Bessel-corrected (n-1) by default in MATLAB
    z = norminv(confidence);
    h = holding_period;

    v = -(mu * h - z * sigma * sqrt(h));
    cv = -(mu * h - sigma * sqrt(h) * normpdf(z) / (1 - confidence));
end
