% TEST_VAR  Script-based test suite for the MATLAB VaR implementation.
%
% Run with:
%   cd reference/matlab
%   run('test_var.m')
%
% Prints "All tests passed." on success. Calls error() on failure.

this_dir     = fileparts(mfilename('fullpath'));
fixtures_dir = fullfile(this_dir, '..', 'fixtures');

returns = csvread(fullfile(fixtures_dir, 'sample_returns.csv'), 1, 0);
fprintf('Loaded %d returns\n\n', length(returns));

% ---------------------------------------------------------------------------
% Historical VaR
% ---------------------------------------------------------------------------

fprintf('Test 1: historical VaR is positive ... ');
[v, ~] = var_historical(returns);
assert(v > 0);
fprintf('PASS\n');

fprintf('Test 2: historical VaR monotonic in confidence ... ');
[v90, ~] = var_historical(returns, 0.90);
[v95, ~] = var_historical(returns, 0.95);
[v99, ~] = var_historical(returns, 0.99);
assert(v90 < v95 && v95 < v99);
fprintf('PASS\n');

fprintf('Test 3: historical CVaR >= VaR ... ');
for conf = [0.90, 0.95, 0.99]
    [v, cv] = var_historical(returns, conf);
    assert(cv >= v, 'CVaR < VaR at conf=%g', conf);
end
fprintf('PASS\n');

% ---------------------------------------------------------------------------
% Parametric VaR
% ---------------------------------------------------------------------------

fprintf('Test 4: parametric VaR is positive ... ');
[v, ~] = var_parametric(returns);
assert(v > 0);
fprintf('PASS\n');

fprintf('Test 5: parametric VaR monotonic in confidence ... ');
[v90, ~] = var_parametric(returns, 0.90);
[v95, ~] = var_parametric(returns, 0.95);
[v99, ~] = var_parametric(returns, 0.99);
assert(v90 < v95 && v95 < v99);
fprintf('PASS\n');

fprintf('Test 6: parametric CVaR >= VaR ... ');
for conf = [0.90, 0.95, 0.99]
    [v, cv] = var_parametric(returns, conf);
    assert(cv >= v, 'CVaR < VaR at conf=%g', conf);
end
fprintf('PASS\n');

% ---------------------------------------------------------------------------
% Monte Carlo VaR
% ---------------------------------------------------------------------------

fprintf('Test 7: MC VaR is positive ... ');
mc = var_monte_carlo(returns);
assert(mc.var > 0);
fprintf('PASS\n');

fprintf('Test 8: MC CVaR >= VaR ... ');
assert(mc.cvar >= mc.var);
fprintf('PASS\n');

fprintf('Test 9: MC seed reproducibility ... ');
mc1 = var_monte_carlo(returns, 0.95, 1, 10000, 123);
mc2 = var_monte_carlo(returns, 0.95, 1, 10000, 123);
assert(mc1.var == mc2.var);
assert(mc1.cvar == mc2.cvar);
fprintf('PASS\n');

% ---------------------------------------------------------------------------
% Fixture validation - historical
% ---------------------------------------------------------------------------

fprintf('Test 10: historical matches fixture values ... ');
expected = readtable(fullfile(fixtures_dir, 'expected_historical.csv'));
for i = 1:height(expected)
    [v, cv] = var_historical(returns, expected.confidence(i), expected.holding_period(i));
    assert(abs(v - expected.var(i)) / expected.var(i) < 0.01, ...
        'hist VaR mismatch at row %d', i);
    assert(abs(cv - expected.cvar(i)) / expected.cvar(i) < 0.01, ...
        'hist CVaR mismatch at row %d', i);
end
fprintf('PASS\n');

% ---------------------------------------------------------------------------
% Fixture validation - parametric
% ---------------------------------------------------------------------------

fprintf('Test 11: parametric matches fixture values ... ');
expected = readtable(fullfile(fixtures_dir, 'expected_parametric.csv'));
for i = 1:height(expected)
    [v, cv] = var_parametric(returns, expected.confidence(i), expected.holding_period(i));
    assert(abs(v - expected.var(i)) / expected.var(i) < 0.001, ...
        'param VaR mismatch at row %d', i);
    assert(abs(cv - expected.cvar(i)) / expected.cvar(i) < 0.001, ...
        'param CVaR mismatch at row %d', i);
end
fprintf('PASS\n');

fprintf('\n11/11 tests passed\n');
