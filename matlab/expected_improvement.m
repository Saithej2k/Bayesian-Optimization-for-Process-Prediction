function ei = expected_improvement(mu, sigma, bestObserved, xi)
%EXPECTED_IMPROVEMENT Vectorized EI for maximization.

if nargin < 4
    xi = 0.01;
end

sigma = max(sigma, 0);
improvement = mu - bestObserved - xi;
ei = zeros(size(mu));
idx = sigma > 1e-12;
z = improvement(idx) ./ sigma(idx);
ei(idx) = improvement(idx) .* normcdf(z) + sigma(idx) .* normpdf(z);
ei = max(ei, 0);
end
