function y = process_surface(x, noiseStd, seed)
%PROCESS_SURFACE Noisy process-yield surface matching the Python objective.
%   x is an N-by-2 matrix: [temperature_c, residence_time_min].

if nargin < 2 || isempty(noiseStd)
    noiseStd = 0;
end
if nargin >= 3 && ~isempty(seed)
    rng(seed);
end

lower = [120, 5];
upper = [220, 45];
z = (x - lower) ./ (upper - lower);
temperature = z(:, 1);
time = z(:, 2);

primary = 41.0 .* exp(-(((temperature - 0.58) ./ 0.16) .^ 2 + ((time - 0.58) ./ 0.19) .^ 2));
secondary = 12.0 .* exp(-(((temperature - 0.27) ./ 0.14) .^ 2 + ((time - 0.26) ./ 0.12) .^ 2));
ridge = 9.0 .* exp(-((temperature - 0.42) ./ 0.32) .^ 2) .* exp(-((time - 0.70) ./ 0.28) .^ 2);
disturbance = 2.4 .* sin(3.6 .* pi .* temperature) .* cos(2.3 .* pi .* time);
harshPenalty = 16.0 .* max(temperature - 0.77, 0.0) .^ 2 + 10.0 .* max(time - 0.82, 0.0) .^ 2;
lowConversionPenalty = 6.0 .* max(0.18 - time, 0.0) .^ 2;

y = 51.0 + primary + secondary + ridge + disturbance - harshPenalty - lowConversionPenalty;
y = min(max(y, 0.0), 100.0);

if noiseStd > 0
    y = y + noiseStd .* randn(size(y));
end
end
