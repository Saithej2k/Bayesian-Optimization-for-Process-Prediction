% BAYESOPT_DEMO Gaussian-process Bayesian optimization demo for process yield.
clear; clc;

bounds = [120, 220; 5, 45];
seed = 7;
noiseStd = 1.2;
budget = 45;
nInitial = 8;
nCandidates = 2500;
rng(seed);

X = lhsdesign(nInitial, 2) .* (bounds(:, 2)' - bounds(:, 1)') + bounds(:, 1)';
y = process_surface(X, noiseStd);

for iter = (nInitial + 1):budget
    gp = fitrgp(X, y, ...
        'KernelFunction', 'matern52', ...
        'Standardize', true, ...
        'Sigma', noiseStd);

    candidates = rand(nCandidates, 2) .* (bounds(:, 2)' - bounds(:, 1)') + bounds(:, 1)';
    [mu, sigma] = predict(gp, candidates);
    ei = expected_improvement(mu, sigma, max(y), 0.05);
    [~, idx] = max(ei);

    nextX = candidates(idx, :);
    nextY = process_surface(nextX, noiseStd);
    X = [X; nextX]; %#ok<AGROW>
    y = [y; nextY]; %#ok<AGROW>
end

trueY = process_surface(X, 0);
[bestY, bestIdx] = max(trueY);
fprintf('Best true yield: %.2f%% at %.2f C, %.2f min\n', bestY, X(bestIdx, 1), X(bestIdx, 2));

[tempGrid, timeGrid] = meshgrid(linspace(bounds(1, 1), bounds(1, 2), 100), linspace(bounds(2, 1), bounds(2, 2), 100));
gridX = [tempGrid(:), timeGrid(:)];
gridY = reshape(process_surface(gridX, 0), size(tempGrid));

figure;
contourf(tempGrid, timeGrid, gridY, 32, 'LineStyle', 'none');
colorbar;
hold on;
plot(X(:, 1), X(:, 2), 'w.-', 'LineWidth', 1.2, 'MarkerSize', 10);
plot(X(bestIdx, 1), X(bestIdx, 2), 'ro', 'MarkerFaceColor', 'r');
xlabel('Temperature (C)');
ylabel('Residence time (min)');
title('Bayesian optimization samples on process-yield surface');
