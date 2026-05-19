1;

% Task 4. Optimal portfolio in the Tobin-Sharpe-Lintner model.
% Octave-compatible version. It uses only base Octave functions:
% csvread, qp, fprintf and print. No extra Octave packages are required.
%
% Data safety:
% - Data_zad4_2026.xlsx is not modified.
% - prepare_data.py exports lossless CSV copies to prepared/.

function [A, b] = twoSidedLinearConstraints(G, lowerBound, upperBound)
    A = [G; -G];
    b = [upperBound; -lowerBound];
endfunction

function [A, b] = groupComparisonConstraints(G1, G2, lowerBound, upperBound)
    m = rows(G1);
    A = zeros(2 * m, columns(G1));
    b = zeros(2 * m, 1);
    for i = 1:m
        A(2*i-1, :) = -G1(i, :) + lowerBound(i) * G2(i, :);
        A(2*i, :) = G1(i, :) - upperBound(i) * G2(i, :);
    endfor
endfunction

function x0 = initialPoint(A, b, lb, ub)
    n = numel(lb);
    candidates = [
        ones(n, 1) / n, ...
        [0.10; 0.15; 0.15; 0.30; 0.15; 0.15], ...
        [0.05; 0.30; 0.0633333333; 0.27; 0.39; 0.1266666667]
    ];
    for k = 1:columns(candidates)
        x = candidates(:, k);
        if all(x >= lb - 1e-10) && all(x <= ub + 1e-10)
            if isempty(A) || all(A * x <= b + 1e-10)
                x0 = x;
                return;
            endif
        endif
    endfor
    x0 = (lb + ub) / 2;
endfunction

function [x, ok] = solveQp(x0, H, q, Aeq, beq, lb, ub, Aineq, bineq)
    if isempty(Aineq)
        A_lb = [];
        A_in = [];
        A_ub = [];
    else
        A_lb = -Inf(rows(Aineq), 1);
        A_in = Aineq;
        A_ub = bineq;
    endif

    options = struct('MaxIter', 2000, 'TolX', 1e-10, 'AllowSemidefinite', true);
    H = (H + H') / 2 + 1e-10 * eye(rows(H));
    [x, ~, info] = qp(x0, H, q, Aeq, beq, lb, ub, A_lb, A_in, A_ub, options);
    ok = isstruct(info) && (info.info == 0 || info.info == 1);
endfunction

function out = solveLinearReturn(mu, lb, ub, Aineq, bineq, Aeq, beq, minimizeReturn)
    n = numel(mu);
    x0 = initialPoint(Aineq, bineq, lb, ub);
    if minimizeReturn
        q = mu;
    else
        q = -mu;
    endif
    [x, ok] = solveQp(x0, 1e-8 * eye(n), q, Aeq, beq, lb, ub, Aineq, bineq);
    if ~ok
        error('Could not find a feasible return bound.');
    endif
    out = mu' * x;
endfunction

function [riskVals, returnVals, weightVals] = computeFrontier(mu, Sigma, lb, ub, Aineq, bineq, nPoints)
    n = numel(mu);
    x0 = initialPoint(Aineq, bineq, lb, ub);
    AeqSum = ones(1, n);
    beqSum = 1;

    minReturn = solveLinearReturn(mu, lb, ub, Aineq, bineq, AeqSum, beqSum, true);
    maxReturn = solveLinearReturn(mu, lb, ub, Aineq, bineq, AeqSum, beqSum, false);
    targets = linspace(minReturn, maxReturn, nPoints);

    riskVals = [];
    returnVals = [];
    weightVals = [];

    for k = 1:nPoints
        Aeq = [AeqSum; mu'];
        beq = [beqSum; targets(k)];
        [x, ok] = solveQp(x0, Sigma, zeros(n, 1), Aeq, beq, lb, ub, Aineq, bineq);
        if ok
            riskVals(end + 1, 1) = sqrt(x' * Sigma * x);
            returnVals(end + 1, 1) = mu' * x;
            weightVals(end + 1, :) = x';
            x0 = x;
        endif
    endfor
endfunction

function sol = solveTslCase(mu, Sigma, rf, theta, lb, ub, Aineq, bineq, caseName)
    n = numel(mu);
    x0 = initialPoint(Aineq, bineq, lb, ub);
    H = theta * Sigma;
    q = -(mu - rf);
    [x, ok] = solveQp(x0, H, q, [], [], lb, ub, Aineq, bineq);

    if ~ok
        sol.weights = nan(n, 1);
        sol.expectedReturn = -Inf;
        sol.risk = Inf;
        sol.utility = -Inf;
        sol.caseName = caseName;
        return;
    endif

    xi = sum(x);
    riskFreeWeight = 1 - xi;
    expectedReturn = rf * riskFreeWeight + mu' * x;
    risk = sqrt(x' * Sigma * x);
    utility = expectedReturn - 0.5 * theta * risk^2;

    sol.weights = x;
    sol.expectedReturn = expectedReturn;
    sol.risk = risk;
    sol.utility = utility;
    sol.caseName = caseName;
endfunction

function sol = solveTslPortfolio(mu, Sigma, rfLend, rfBorrow, theta, lb, ub, Aineq, bineq)
    n = numel(mu);
    lendA = [Aineq; ones(1, n)];
    lendB = [bineq; 1];
    borrowA = [Aineq; -ones(1, n)];
    borrowB = [bineq; -1];

    lend = solveTslCase(mu, Sigma, rfLend, theta, lb, ub, lendA, lendB, 'lending');
    borrow = solveTslCase(mu, Sigma, rfBorrow, theta, lb, ub, borrowA, borrowB, 'borrowing');

    if lend.utility >= borrow.utility
        sol = lend;
    else
        sol = borrow;
    endif
endfunction

function ratios = safeRatios(numerators, denominators)
    ratios = nan(size(numerators));
    for i = 1:numel(numerators)
        if abs(denominators(i)) > 1e-12
            ratios(i) = numerators(i) / denominators(i);
        endif
    endfor
endfunction

function rowsOut = appendActivityRows(rowsIn, scenario, theta, assets, x, lb, ub, ...
    groupNames, groupValues, gl, gu, comparisonNames, comparisonValues, cl, cu)
    rowsOut = rowsIn;
    tol = 1e-5;

    for i = 1:numel(assets)
        rowsOut(end + 1, :) = {scenario, theta, 'Asset', assets{i}, ...
            x(i), lb(i), ub(i), abs(x(i) - lb(i)) <= tol, abs(x(i) - ub(i)) <= tol};
    endfor

    for i = 1:numel(groupNames)
        rowsOut(end + 1, :) = {scenario, theta, 'Group', groupNames{i}, ...
            groupValues(i), gl(i), gu(i), abs(groupValues(i) - gl(i)) <= tol, abs(groupValues(i) - gu(i)) <= tol};
    endfor

    for i = 1:numel(comparisonNames)
        rowsOut(end + 1, :) = {scenario, theta, 'GroupComparison', comparisonNames{i}, ...
            comparisonValues(i), cl(i), cu(i), abs(comparisonValues(i) - cl(i)) <= tol, abs(comparisonValues(i) - cu(i)) <= tol};
    endfor
endfunction

function writeAssetEstimates(path, assets, muDaily, muAnnual, riskAnnual)
    fid = fopen(path, 'w');
    fprintf(fid, 'Ticker,ExpectedDailyReturn,ExpectedAnnualReturn,AnnualVolatility\n');
    for i = 1:numel(assets)
        fprintf(fid, '%s,%.15g,%.15g,%.15g\n', assets{i}, muDaily(i), muAnnual(i), riskAnnual(i));
    endfor
    fclose(fid);
endfunction

function writeNamedMatrix(path, names, M)
    fid = fopen(path, 'w');
    fprintf(fid, 'Asset');
    for i = 1:numel(names)
        fprintf(fid, ',%s', names{i});
    endfor
    fprintf(fid, '\n');
    for i = 1:rows(M)
        fprintf(fid, '%s', names{i});
        for j = 1:columns(M)
            fprintf(fid, ',%.15g', M(i, j));
        endfor
        fprintf(fid, '\n');
    endfor
    fclose(fid);
endfunction

function writeFrontier(path, assets, scenarios, rowsData)
    fid = fopen(path, 'w');
    fprintf(fid, 'Scenario,RiskAnnual,ReturnAnnual');
    for i = 1:numel(assets)
        fprintf(fid, ',%s', assets{i});
    endfor
    fprintf(fid, '\n');
    for i = 1:rows(rowsData)
        fprintf(fid, '%s,%.15g,%.15g', scenarios{i}, rowsData(i, 1), rowsData(i, 2));
        for j = 3:columns(rowsData)
            fprintf(fid, ',%.15g', rowsData(i, j));
        endfor
        fprintf(fid, '\n');
    endfor
    fclose(fid);
endfunction

function writeOptimal(path, assets, scenarios, cases, rowsData)
    fid = fopen(path, 'w');
    fprintf(fid, 'Scenario,Theta,RiskFreeCase,ExpectedReturnAnnual,RiskAnnual,Utility,XiRiskyAllocation,RiskFreeWeight');
    for i = 1:numel(assets)
        fprintf(fid, ',%s', assets{i});
    endfor
    fprintf(fid, '\n');
    for i = 1:rows(rowsData)
        fprintf(fid, '%s,%.15g,%s', scenarios{i}, rowsData(i, 1), cases{i});
        for j = 2:columns(rowsData)
            fprintf(fid, ',%.15g', rowsData(i, j));
        endfor
        fprintf(fid, '\n');
    endfor
    fclose(fid);
endfunction

function writeActivity(path, activityRows)
    fid = fopen(path, 'w');
    fprintf(fid, 'Scenario,Theta,ConstraintType,Name,Value,LowerBound,UpperBound,ActiveLower,ActiveUpper\n');
    for i = 1:rows(activityRows)
        fprintf(fid, '%s,%.15g,%s,%s,%.15g,%.15g,%.15g,%d,%d\n', ...
            activityRows{i, 1}, activityRows{i, 2}, activityRows{i, 3}, activityRows{i, 4}, ...
            activityRows{i, 5}, activityRows{i, 6}, activityRows{i, 7}, activityRows{i, 8}, activityRows{i, 9});
    endfor
    fclose(fid);
endfunction

%% Main calculation
clc; close all;

rootDir = fileparts(mfilename('fullpath'));
preparedDir = fullfile(rootDir, 'prepared');
resultsDir = fullfile(rootDir, 'results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
endif

returnsFile = fullfile(preparedDir, 'returns_simple.csv');
if ~exist(returnsFile, 'file')
    error('Missing prepared/returns_simple.csv. Run: python Finance/4/prepare_data.py');
endif

assets = {'GAZP', 'ROSN', 'LKOH', 'FEES', 'SBER', 'VTBR'};
n = numel(assets);
tradingDaysPerYear = 252;
rfLendAnnual = 0.0775;
rfBorrowAnnual = rfLendAnnual + 0.05;
thetaValues = [2 3 4 8 12 20 40];

R = csvread(returnsFile, 1, 1);
muDaily = mean(R, 1)';
covDaily = cov(R);
mu = muDaily * tradingDaysPerYear;
Sigma = covDaily * tradingDaysPerYear;
assetRisk = sqrt(diag(Sigma));

writeAssetEstimates(fullfile(resultsDir, 'asset_estimates.csv'), assets, muDaily, mu, assetRisk);
writeNamedMatrix(fullfile(resultsDir, 'covariance_annual.csv'), assets, Sigma);

fprintf('\nAnnualized estimates:\n');
fprintf('%-8s %14s %14s\n', 'Asset', 'Return', 'Risk');
for i = 1:n
    fprintf('%-8s %13.4f%% %13.4f%%\n', assets{i}, 100 * mu(i), 100 * assetRisk(i));
endfor

lb = 0.05 * ones(n, 1);
ub = 0.39 * ones(n, 1);

GA = [
    1 1 1 0 0 0
    0 0 0 1 0 0
    0 0 0 0 1 1
    0 0 1 1 1 1
    1 1 0 0 0 0
];
groupNames = {'OilGas', 'Energy', 'Banks', 'Internal', 'External'};
gl = [0.25; 0.27; 0.15; 0.25; 0.10];
gu = [0.65; 0.75; 0.55; 0.85; 0.35];
[Agroup, bgroup] = twoSidedLinearConstraints(GA, gl, gu);

G1 = [
    0 0 1 1 1 1
    1 1 1 0 0 0
];
G2 = [
    1 1 0 0 0 0
    0 0 0 0 1 1
];
comparisonNames = {'InternalToExternal', 'OilGasToBanks'};
comparisonLower = [1.0; 0.8];
comparisonUpper = [5.0; 3.0];
[Acomparison, bcomparison] = groupComparisonConstraints(G1, G2, comparisonLower, comparisonUpper);

scenarios = struct([]);
scenarios(1).name = 'Asset limits only';
scenarios(1).code = 'asset_limits';
scenarios(1).A = [];
scenarios(1).b = [];

scenarios(2).name = 'Asset + group limits';
scenarios(2).code = 'group_limits';
scenarios(2).A = Agroup;
scenarios(2).b = bgroup;

scenarios(3).name = 'Asset + group + comparison';
scenarios(3).code = 'group_comparison';
scenarios(3).A = [Agroup; Acomparison];
scenarios(3).b = [bgroup; bcomparison];

frontierRows = [];
frontierScenario = {};

figure('visible', 'off');
hold on; grid on;
colors = lines(numel(scenarios));

for s = 1:numel(scenarios)
    [riskVals, returnVals, weightVals] = computeFrontier(mu, Sigma, lb, ub, scenarios(s).A, scenarios(s).b, 35);
    plot(riskVals, returnVals, 'LineWidth', 2.2, 'Color', colors(s, :));

    for k = 1:numel(riskVals)
        frontierScenario{end + 1, 1} = scenarios(s).code;
        frontierRows(end + 1, :) = [riskVals(k), returnVals(k), weightVals(k, :)];
    endfor
endfor

xlabel('Annual risk, sigma');
ylabel('Expected annual return');
title('Efficient frontiers under different constraints');
legend({scenarios.name}, 'Location', 'best');
print(gcf, fullfile(resultsDir, 'efficient_frontiers.png'), '-dpng', '-r180');
close(gcf);

writeFrontier(fullfile(resultsDir, 'frontier_points.csv'), assets, frontierScenario, frontierRows);

optimalRows = [];
optimalScenario = {};
optimalCase = {};
activityRows = {};

fprintf('\nOptimal portfolios:\n');
fprintf('%-18s %5s %-10s %10s %10s %8s\n', 'Scenario', 'theta', 'case', 'return', 'risk', 'xi');

for s = 1:numel(scenarios)
    for theta = thetaValues
        sol = solveTslPortfolio(mu, Sigma, rfLendAnnual, rfBorrowAnnual, theta, ...
            lb, ub, scenarios(s).A, scenarios(s).b);

        x = sol.weights;
        xi = sum(x);
        riskFreeWeight = 1 - xi;
        groupValues = GA * x;
        comparisonValues = safeRatios(G1 * x, G2 * x);

        optimalScenario{end + 1, 1} = scenarios(s).code;
        optimalCase{end + 1, 1} = sol.caseName;
        optimalRows(end + 1, :) = [theta, sol.expectedReturn, sol.risk, sol.utility, xi, riskFreeWeight, x'];

        activityRows = appendActivityRows(activityRows, scenarios(s).code, theta, ...
            assets, x, lb, ub, groupNames, groupValues, gl, gu, ...
            comparisonNames, comparisonValues, comparisonLower, comparisonUpper);

        fprintf('%-18s %5.0f %-10s %9.2f%% %9.2f%% %8.3f\n', ...
            scenarios(s).code, theta, sol.caseName, 100 * sol.expectedReturn, 100 * sol.risk, xi);
    endfor
endfor

writeOptimal(fullfile(resultsDir, 'optimal_portfolios.csv'), assets, optimalScenario, optimalCase, optimalRows);
writeActivity(fullfile(resultsDir, 'constraint_activity.csv'), activityRows);

fprintf('\nInterpretation of xi:\n');
fprintf('xi = sum of risky-asset weights, i.e. allocation to the tangent risky portfolio.\n');
fprintf('xi < 1: the remainder is invested in the risk-free asset.\n');
fprintf('xi > 1: borrowing is used to increase the risky allocation.\n');
fprintf('xi = 1: no risk-free asset is used.\n\n');
