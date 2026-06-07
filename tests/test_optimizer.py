from process_bo.experiment import run_benchmark
from process_bo.objective import NoisyProcessObjective, ProcessBounds
from process_bo.optimizer import BayesianOptimizer


def test_bayesian_optimizer_reaches_high_yield_region():
    bounds = ProcessBounds.default()
    objective = NoisyProcessObjective(seed=7, noise_std=0.5, bounds=bounds)
    optimizer = BayesianOptimizer(
        objective.evaluate,
        bounds.array,
        true_objective=objective.true,
        noise_std=0.5,
        seed=7,
        n_candidates=1024,
    )

    trace = optimizer.run(acquisition="ei", n_initial=6, budget=24)

    assert trace.n_evaluations == 24
    assert trace.best_y > 84.0
    assert bounds.lower[0] <= trace.best_x[0] <= bounds.upper[0]
    assert bounds.lower[1] <= trace.best_x[1] <= bounds.upper[1]


def test_benchmark_reports_at_least_60_percent_reduction(tmp_path):
    summary = run_benchmark(seed=7, budget=60, grid_size=25, output_dir=tmp_path)
    reductions = [row["evaluation_reduction_vs_grid"] for row in summary["results"]]

    assert all(row["reached_target"] for row in summary["results"])
    assert all(reduction >= 0.60 for reduction in reductions)
    assert (tmp_path / "summary.json").exists()
