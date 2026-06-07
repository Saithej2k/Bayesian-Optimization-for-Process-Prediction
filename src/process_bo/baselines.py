"""Baseline search strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from process_bo.sampling import grid_candidates


@dataclass
class GridSearchResult:
    x: np.ndarray
    y_observed: np.ndarray
    y_true: np.ndarray | None
    best_observed: np.ndarray
    best_true: np.ndarray | None

    def evaluations_to_target(self, target: float, *, use_true: bool = True) -> int | None:
        history = self.best_true if use_true and self.best_true is not None else self.best_observed
        hits = np.flatnonzero(history >= target)
        if hits.size == 0:
            return None
        return int(hits[0] + 1)


def run_grid_search(
    objective: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    *,
    grid_size: int = 25,
    true_objective: Callable[[np.ndarray], np.ndarray] | None = None,
) -> GridSearchResult:
    candidates = grid_candidates(bounds, grid_size)
    y_observed = objective(candidates).astype(float)
    y_true = true_objective(candidates).astype(float) if true_objective else None
    return GridSearchResult(
        x=candidates,
        y_observed=y_observed,
        y_true=y_true,
        best_observed=np.maximum.accumulate(y_observed),
        best_true=np.maximum.accumulate(y_true) if y_true is not None else None,
    )
