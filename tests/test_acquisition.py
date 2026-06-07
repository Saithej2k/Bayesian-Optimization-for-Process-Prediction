import numpy as np

from process_bo.acquisition import expected_improvement, upper_confidence_bound


def test_expected_improvement_prefers_higher_promising_mean():
    mean = np.array([5.0, 8.0, 9.0])
    std = np.array([0.5, 0.5, 0.5])
    ei = expected_improvement(mean, std, best_observed=7.0, xi=0.0)

    assert ei[2] > ei[1] > ei[0]
    assert np.all(ei >= 0.0)


def test_expected_improvement_zero_for_zero_uncertainty_without_improvement():
    ei = expected_improvement(np.array([3.0]), np.array([0.0]), best_observed=5.0)

    assert ei.item() == 0.0


def test_upper_confidence_bound_includes_uncertainty_bonus():
    mean = np.array([10.0, 10.0])
    std = np.array([0.1, 2.0])
    ucb = upper_confidence_bound(mean, std, kappa=2.0)

    assert ucb[1] > ucb[0]
