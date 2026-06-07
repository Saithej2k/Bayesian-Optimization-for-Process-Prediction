"""Acquisition functions for Gaussian-process Bayesian optimization."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _as_arrays(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_arr = np.asarray(mean, dtype=float)
    std_arr = np.asarray(std, dtype=float)
    return mean_arr, np.maximum(std_arr, 0.0)


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
    *,
    xi: float = 0.01,
    maximize: bool = True,
) -> np.ndarray:
    """Return the Expected Improvement acquisition value.

    Parameters are vectorized over candidate points. For maximization, EI is
    E[max(f(x) - best_observed - xi, 0)]. For minimization, the sign of the
    improvement is inverted.
    """

    mean_arr, std_arr = _as_arrays(mean, std)
    improvement = mean_arr - best_observed - xi
    if not maximize:
        improvement = best_observed - mean_arr - xi

    ei = np.zeros_like(mean_arr, dtype=float)
    nonzero = std_arr > 1e-12
    if np.any(nonzero):
        z = improvement[nonzero] / std_arr[nonzero]
        ei[nonzero] = improvement[nonzero] * norm.cdf(z) + std_arr[nonzero] * norm.pdf(z)
    return np.maximum(ei, 0.0)


def upper_confidence_bound(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    kappa: float = 2.0,
    maximize: bool = True,
) -> np.ndarray:
    """Return the Gaussian-process UCB acquisition value."""

    mean_arr, std_arr = _as_arrays(mean, std)
    if maximize:
        return mean_arr + kappa * std_arr
    return -(mean_arr - kappa * std_arr)


def acquisition_values(
    kind: str,
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
    *,
    xi: float = 0.01,
    kappa: float = 2.0,
    maximize: bool = True,
) -> np.ndarray:
    """Dispatch by acquisition name."""

    normalized = kind.strip().lower()
    if normalized in {"ei", "expected_improvement", "expected-improvement"}:
        return expected_improvement(mean, std, best_observed, xi=xi, maximize=maximize)
    if normalized in {"ucb", "upper_confidence_bound", "upper-confidence-bound"}:
        return upper_confidence_bound(mean, std, kappa=kappa, maximize=maximize)
    raise ValueError(f"Unknown acquisition function: {kind!r}")
