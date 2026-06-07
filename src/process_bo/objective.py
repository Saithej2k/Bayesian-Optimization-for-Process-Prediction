"""Noisy process-yield objective used by the BO experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ProcessBounds:
    """Physical operating bounds for the process controls."""

    lower: np.ndarray
    upper: np.ndarray
    names: tuple[str, ...] = ("temperature_c", "residence_time_min")

    @classmethod
    def default(cls) -> "ProcessBounds":
        return cls(
            lower=np.array([120.0, 5.0], dtype=float),
            upper=np.array([220.0, 45.0], dtype=float),
        )

    @property
    def array(self) -> np.ndarray:
        return np.column_stack([self.lower, self.upper])

    @property
    def dimension(self) -> int:
        return int(self.lower.size)

    def scale_unit(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return (x_arr - self.lower) / (self.upper - self.lower)

    def unscale_unit(self, z: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, dtype=float)
        return self.lower + z_arr * (self.upper - self.lower)


def _ensure_2d(x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Process input must have shape (n_samples, 2).")
    return arr


def process_yield(x: np.ndarray | Sequence[Sequence[float]], bounds: ProcessBounds | None = None) -> np.ndarray:
    """Noise-free process yield in percent for temperature/time controls.

    The surface is deliberately realistic enough for BO:
    - a broad primary operating window,
    - a smaller secondary mode,
    - mild oscillatory process disturbance,
    - penalties near aggressive high-temperature/long-time operation.
    """

    process_bounds = bounds or ProcessBounds.default()
    x_arr = _ensure_2d(x)
    z = process_bounds.scale_unit(x_arr)
    temperature = z[:, 0]
    time = z[:, 1]

    primary = 41.0 * np.exp(-(((temperature - 0.58) / 0.16) ** 2 + ((time - 0.58) / 0.19) ** 2))
    secondary = 12.0 * np.exp(-(((temperature - 0.27) / 0.14) ** 2 + ((time - 0.26) / 0.12) ** 2))
    ridge = 9.0 * np.exp(-((temperature - 0.42) / 0.32) ** 2) * np.exp(-((time - 0.70) / 0.28) ** 2)
    disturbance = 2.4 * np.sin(3.6 * np.pi * temperature) * np.cos(2.3 * np.pi * time)
    harsh_penalty = 16.0 * np.maximum(temperature - 0.77, 0.0) ** 2 + 10.0 * np.maximum(time - 0.82, 0.0) ** 2
    low_conversion_penalty = 6.0 * np.maximum(0.18 - time, 0.0) ** 2

    yield_percent = 51.0 + primary + secondary + ridge + disturbance - harsh_penalty - low_conversion_penalty
    return np.clip(yield_percent, 0.0, 100.0)


@dataclass
class NoisyProcessObjective:
    """Callable noisy measurement wrapper around the process-yield surface."""

    noise_std: float = 1.2
    seed: int = 7
    bounds: ProcessBounds = ProcessBounds.default()

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def true(self, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        return process_yield(x, self.bounds)

    def evaluate(self, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        x_arr = _ensure_2d(x)
        noise = self.rng.normal(loc=0.0, scale=self.noise_std, size=x_arr.shape[0])
        return self.true(x_arr) + noise
