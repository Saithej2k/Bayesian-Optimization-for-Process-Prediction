"""Plotting helpers for experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from process_bo.objective import ProcessBounds, process_yield
from process_bo.sampling import grid_candidates


def plot_surface_with_samples(
    bounds: ProcessBounds,
    samples: np.ndarray,
    output_path: Path,
    *,
    grid_size: int = 120,
    title: str = "Process yield surface",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = grid_candidates(bounds.array, grid_size)
    z = process_yield(grid, bounds).reshape(grid_size, grid_size)
    x_axis = np.linspace(bounds.lower[0], bounds.upper[0], grid_size)
    y_axis = np.linspace(bounds.lower[1], bounds.upper[1], grid_size)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    contour = ax.contourf(x_axis, y_axis, z.T, levels=32, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="Yield (%)")
    ax.plot(samples[:, 0], samples[:, 1], "w.-", linewidth=1.0, markersize=4, label="BO samples")
    ax.scatter(samples[-1, 0], samples[-1, 1], c="#ff5a5f", s=70, edgecolor="black", label="Final sample")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Residence time (min)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_convergence(series: dict[str, np.ndarray], output_path: Path, *, target: float | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, values in series.items():
        steps = np.arange(1, len(values) + 1)
        ax.plot(steps, values, linewidth=2, label=label)
    if target is not None:
        ax.axhline(target, color="#d95f02", linestyle="--", linewidth=1.4, label="target")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best true yield (%)")
    ax.set_title("Optimization convergence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
