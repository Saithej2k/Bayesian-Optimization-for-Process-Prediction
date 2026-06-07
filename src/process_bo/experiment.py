"""Benchmark EI/UCB Bayesian optimization against grid search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from process_bo.baselines import run_grid_search
from process_bo.objective import NoisyProcessObjective, ProcessBounds, process_yield
from process_bo.optimizer import BayesianOptimizer, OptimizationTrace
from process_bo.sampling import grid_candidates
from process_bo.visualization import plot_convergence, plot_surface_with_samples


def estimate_true_optimum(bounds: ProcessBounds, grid_size: int = 180) -> tuple[np.ndarray, float]:
    grid = grid_candidates(bounds.array, grid_size)
    y = process_yield(grid, bounds)
    idx = int(np.argmax(y))
    return grid[idx], float(y[idx])


def trace_to_frame(trace: OptimizationTrace, bounds: ProcessBounds) -> pd.DataFrame:
    frame = pd.DataFrame(trace.x, columns=bounds.names)
    frame["y_observed"] = trace.y_observed
    if trace.y_true is not None:
        frame["y_true"] = trace.y_true
        frame["best_true"] = trace.best_true
    frame["best_observed"] = trace.best_observed
    frame["acquisition"] = trace.acquisition
    frame["evaluation"] = np.arange(1, trace.n_evaluations + 1)
    return frame


def run_benchmark(
    *,
    seed: int = 7,
    budget: int = 60,
    n_initial: int = 8,
    noise_std: float = 1.2,
    grid_size: int = 25,
    target_margin: float = 1.0,
    output_dir: str | Path = "artifacts",
) -> dict[str, object]:
    bounds = ProcessBounds.default()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    optimum_x, optimum_y = estimate_true_optimum(bounds)
    target_y = optimum_y - target_margin

    traces: dict[str, OptimizationTrace] = {}
    for offset, acquisition in enumerate(("ei", "ucb")):
        objective = NoisyProcessObjective(noise_std=noise_std, seed=seed + offset, bounds=bounds)
        optimizer = BayesianOptimizer(
            objective.evaluate,
            bounds.array,
            true_objective=objective.true,
            noise_std=noise_std,
            seed=seed + offset,
            n_candidates=4096,
        )
        traces[acquisition] = optimizer.run(
            acquisition=acquisition,
            n_initial=n_initial,
            budget=budget,
            xi=0.05,
            kappa=1.8,
        )

    grid_objective = NoisyProcessObjective(noise_std=noise_std, seed=seed + 99, bounds=bounds)
    grid = run_grid_search(
        grid_objective.evaluate,
        bounds.array,
        grid_size=grid_size,
        true_objective=grid_objective.true,
    )

    all_frames = [trace_to_frame(trace, bounds) for trace in traces.values()]
    samples = pd.concat(all_frames, ignore_index=True)
    samples.to_csv(output_path / "bo_samples.csv", index=False)

    grid_frame = pd.DataFrame(grid.x, columns=bounds.names)
    grid_frame["y_observed"] = grid.y_observed
    if grid.y_true is not None:
        grid_frame["y_true"] = grid.y_true
        grid_frame["best_true"] = grid.best_true
    grid_frame["best_observed"] = grid.best_observed
    grid_frame["evaluation"] = np.arange(1, len(grid_frame) + 1)
    grid_frame.to_csv(output_path / "grid_search.csv", index=False)

    grid_to_target = grid.evaluations_to_target(target_y, use_true=True) or int(grid.x.shape[0])
    summary_rows: list[dict[str, object]] = []
    convergence: dict[str, np.ndarray] = {}
    for acquisition, trace in traces.items():
        bo_to_target = trace.evaluations_to_target(target_y, use_true=True)
        reached_target = bo_to_target is not None
        reduction = 1.0 - (bo_to_target / grid_to_target) if reached_target else None
        summary_rows.append(
            {
                "method": acquisition,
                "evaluations_to_target": bo_to_target,
                "grid_evaluations_to_target": grid_to_target,
                "evaluation_reduction_vs_grid": reduction,
                "reached_target": reached_target,
                "best_true_yield": trace.best_y,
                "best_temperature_c": trace.best_x[0],
                "best_residence_time_min": trace.best_x[1],
                "target_yield": target_y,
                "true_optimum_yield": optimum_y,
            }
        )
        convergence[acquisition.upper()] = trace.best_true if trace.best_true is not None else trace.best_observed
        plot_surface_with_samples(
            bounds,
            trace.x,
            output_path / f"surface_{acquisition}.png",
            title=f"Process surface with {acquisition.upper()} samples",
        )

    convergence["Grid search"] = grid.best_true if grid.best_true is not None else grid.best_observed
    plot_convergence(convergence, output_path / "convergence.png", target=target_y)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_path / "summary.csv", index=False)
    summary_payload = {
        "seed": seed,
        "budget": budget,
        "n_initial": n_initial,
        "noise_std": noise_std,
        "grid_size": grid_size,
        "target_margin": target_margin,
        "true_optimum_x": optimum_x.tolist(),
        "true_optimum_yield": optimum_y,
        "target_yield": target_y,
        "results": summary_rows,
    }
    (output_path / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--n-initial", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=1.2)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--target-margin", type=float, default=1.0)
    parser.add_argument("--output-dir", default="artifacts")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_benchmark(
        seed=args.seed,
        budget=args.budget,
        n_initial=args.n_initial,
        noise_std=args.noise_std,
        grid_size=args.grid_size,
        target_margin=args.target_margin,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
