import numpy as np

from grid_world import GridWorld
from monte_carlo import _first_visit_update, first_visit_mc_control


def test_first_visit_updates_repeated_state_action_only_once() -> None:
    q_values = np.zeros((2, 2, 4), dtype=float)
    counts = np.zeros_like(q_values, dtype=np.int64)
    repeated = ((1, 0), 1)
    trajectory = [
        (repeated[0], repeated[1], -1.0),
        ((1, 1), 0, -1.0),
        (repeated[0], repeated[1], 0.0),
    ]

    first_return = _first_visit_update(trajectory, q_values, counts, gamma=1.0)

    assert first_return == -2.0
    assert counts[repeated[0]][repeated[1]] == 1
    assert q_values[repeated[0]][repeated[1]] == -2.0


def test_mc_control_learns_a_short_path_on_a_tiny_grid() -> None:
    env = GridWorld(
        height=2,
        width=2,
        start=(1, 0),
        goal=(0, 1),
        obstacles=frozenset(),
    )
    result = first_visit_mc_control(
        env,
        episodes=2_000,
        epsilon_end=0.05,
        max_steps=50,
        start_state_probability=0.7,
        seed=11,
    )

    assert env.validate_path(result.path)
    assert result.path_length == 2
    assert np.count_nonzero(result.visit_counts) == 12

