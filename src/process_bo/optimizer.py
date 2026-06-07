"""Gaussian-process Bayesian optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from process_bo.acquisition import acquisition_values
from process_bo.sampling import latin_hypercube, scaled_min_dist, sobol_candidates


ArrayFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class OptimizationTrace:
    """Observed samples and best-so-far history from an optimization run."""

    acquisition: str
    x: np.ndarray
    y_observed: np.ndarray
    y_true: np.ndarray | None
    proposed_x: np.ndarray
    best_observed: np.ndarray
    best_true: np.ndarray | None

    @property
    def best_x(self) -> np.ndarray:
        values = self.y_true if self.y_true is not None else self.y_observed
        index = int(np.argmax(values))
        return self.x[index]

    @property
    def best_y(self) -> float:
        values = self.y_true if self.y_true is not None else self.y_observed
        return float(np.max(values))

    @property
    def n_evaluations(self) -> int:
        return int(self.x.shape[0])

    def evaluations_to_target(self, target: float, *, use_true: bool = True) -> int | None:
        history = self.best_true if use_true and self.best_true is not None else self.best_observed
        hits = np.flatnonzero(history >= target)
        if hits.size == 0:
            return None
        return int(hits[0] + 1)


class BayesianOptimizer:
    """Sequential GP Bayesian optimizer for maximization."""

    def __init__(
        self,
        objective: ArrayFunction,
        bounds: np.ndarray,
        *,
        true_objective: ArrayFunction | None = None,
        noise_std: float = 1.2,
        seed: int = 7,
        n_candidates: int = 4096,
        min_scaled_distance: float = 1e-3,
    ) -> None:
        self.objective = objective
        self.true_objective = true_objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.n_candidates = int(n_candidates)
        self.min_scaled_distance = float(min_scaled_distance)

        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (dimension, 2).")

    def run(
        self,
        *,
        acquisition: str,
        n_initial: int = 8,
        budget: int = 60,
        xi: float = 0.01,
        kappa: float = 2.0,
    ) -> OptimizationTrace:
        if budget <= n_initial:
            raise ValueError("budget must be greater than n_initial.")

        x_observed = latin_hypercube(self.bounds, n_initial, seed=self.seed)
        y_observed = self.objective(x_observed).astype(float)
        y_true = self.true_objective(x_observed).astype(float) if self.true_objective else None
        proposed = [x_observed.copy()]

        for step in range(n_initial, budget):
            gp = self._fit_surrogate(x_observed, y_observed)
            candidates = sobol_candidates(self.bounds, self.n_candidates, seed=self.seed + step)
            mean, std = gp.predict(candidates, return_std=True)
            acquisition_score = acquisition_values(
                acquisition,
                mean,
                std,
                float(np.max(y_observed)),
                xi=xi,
                kappa=kappa,
                maximize=True,
            )

            distance = scaled_min_dist(candidates, x_observed, self.bounds)
            acquisition_score = np.where(distance >= self.min_scaled_distance, acquisition_score, -np.inf)
            next_x = candidates[int(np.argmax(acquisition_score))].reshape(1, -1)
            next_y = self.objective(next_x).astype(float)

            x_observed = np.vstack([x_observed, next_x])
            y_observed = np.concatenate([y_observed, next_y])
            if self.true_objective:
                next_true = self.true_objective(next_x).astype(float)
                y_true = np.concatenate([y_true, next_true]) if y_true is not None else next_true
            proposed.append(next_x.copy())

        best_observed = np.maximum.accumulate(y_observed)
        best_true = np.maximum.accumulate(y_true) if y_true is not None else None
        return OptimizationTrace(
            acquisition=acquisition,
            x=x_observed,
            y_observed=y_observed,
            y_true=y_true,
            proposed_x=np.vstack(proposed),
            best_observed=best_observed,
            best_true=best_true,
        )

    def _fit_surrogate(self, x: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        kernel = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(
            length_scale=np.ones(self.bounds.shape[0]),
            length_scale_bounds=(5e-2, 1e2),
            nu=2.5,
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=max(self.noise_std**2, 1e-8),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=self.seed,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gp.fit(x, y)
        return gp
