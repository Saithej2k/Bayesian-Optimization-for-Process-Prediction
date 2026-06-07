"""Sampling helpers for design of experiments and candidate generation."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def latin_hypercube(bounds: np.ndarray, n_samples: int, seed: int | None = None) -> np.ndarray:
    """Draw a Latin-hypercube design inside physical bounds."""

    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError("bounds must have shape (dimension, 2).")
    sampler = qmc.LatinHypercube(d=bounds_arr.shape[0], seed=seed)
    unit = sampler.random(n_samples)
    return qmc.scale(unit, bounds_arr[:, 0], bounds_arr[:, 1])


def sobol_candidates(bounds: np.ndarray, n_samples: int, seed: int | None = None) -> np.ndarray:
    """Draw quasi-random Sobol candidates inside physical bounds."""

    bounds_arr = np.asarray(bounds, dtype=float)
    sampler = qmc.Sobol(d=bounds_arr.shape[0], scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n_samples, 2))))
    unit = sampler.random_base2(m=m)[:n_samples]
    return qmc.scale(unit, bounds_arr[:, 0], bounds_arr[:, 1])


def grid_candidates(bounds: np.ndarray, grid_size: int) -> np.ndarray:
    """Return a dense Cartesian grid in row-major order."""

    bounds_arr = np.asarray(bounds, dtype=float)
    axes = [np.linspace(low, high, grid_size) for low, high in bounds_arr]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([axis.ravel() for axis in mesh])


def scaled_min_dist(x: np.ndarray, observed: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Minimum Euclidean distance from candidates to observations in unit space."""

    x_arr = np.asarray(x, dtype=float)
    observed_arr = np.asarray(observed, dtype=float)
    bounds_arr = np.asarray(bounds, dtype=float)
    span = bounds_arr[:, 1] - bounds_arr[:, 0]
    x_unit = (x_arr - bounds_arr[:, 0]) / span
    observed_unit = (observed_arr - bounds_arr[:, 0]) / span
    deltas = x_unit[:, None, :] - observed_unit[None, :, :]
    return np.sqrt(np.sum(deltas**2, axis=2)).min(axis=1)
