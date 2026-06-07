import numpy as np

from process_bo.objective import NoisyProcessObjective, ProcessBounds, process_yield


def test_process_yield_is_vectorized_and_bounded():
    bounds = ProcessBounds.default()
    x = np.array([[120.0, 5.0], [178.0, 28.0], [220.0, 45.0]])
    y = process_yield(x, bounds)

    assert y.shape == (3,)
    assert np.all((0.0 <= y) & (y <= 100.0))
    assert y[1] > y[0]


def test_noisy_objective_is_reproducible_with_seed():
    x = np.array([[178.0, 28.0], [170.0, 24.0]])
    first = NoisyProcessObjective(seed=123).evaluate(x)
    second = NoisyProcessObjective(seed=123).evaluate(x)

    np.testing.assert_allclose(first, second)
