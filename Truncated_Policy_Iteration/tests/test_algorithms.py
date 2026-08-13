import numpy as np

from algorithms import (
    bellman_optimality_residual,
    policy_iteration,
    truncated_policy_iteration,
    value_iteration,
)
from grid_world import GridWorld


def test_all_algorithms_find_an_equally_short_valid_path() -> None:
    env = GridWorld()
    results = [
        value_iteration(env),
        policy_iteration(env),
        truncated_policy_iteration(env, evaluation_sweeps_per_improvement=3),
    ]

    assert all(env.validate_path(result.path) for result in results)
    assert len({result.path_length for result in results}) == 1
    assert results[0].path_length == 24


def test_all_algorithms_converge_to_the_same_values() -> None:
    env = GridWorld()
    vi = value_iteration(env)
    pi = policy_iteration(env)
    tpi = truncated_policy_iteration(env, evaluation_sweeps_per_improvement=3)

    for result in (pi, tpi):
        for state in env.states:
            assert np.isclose(result.values[state], vi.values[state], atol=1e-6)

    assert bellman_optimality_residual(env, vi.values, gamma=0.95) < 1e-7
    assert bellman_optimality_residual(env, tpi.values, gamma=0.95) < 1e-7
    assert vi.values[env.goal] == 0.0
    assert vi.values[env.start] < 0.0


def test_truncation_parameter_changes_evaluation_schedule() -> None:
    env = GridWorld()
    result = truncated_policy_iteration(env, evaluation_sweeps_per_improvement=5)
    assert result.evaluation_sweeps == result.improvement_steps * 5
