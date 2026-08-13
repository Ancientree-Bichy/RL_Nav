import numpy as np
import pytest

from grid_world import GridWorld


def test_default_grid_has_expected_shape_and_endpoints() -> None:
    env = GridWorld()
    assert (env.height, env.width) == (10, 10)
    assert env.start == (9, 0)
    assert env.goal == (0, 9)
    assert env.start not in env.obstacles
    assert env.goal not in env.obstacles


def test_wall_collision_stays_in_place_and_costs_a_step() -> None:
    env = GridWorld()
    next_state, reward = env.transition((9, 0), action=2)  # down, out of bounds
    assert next_state == (9, 0)
    assert reward == env.step_reward


def test_goal_is_absorbing_and_reward_free() -> None:
    env = GridWorld()
    for action in range(env.n_actions):
        assert env.transition(env.goal, action) == (env.goal, 0.0)


def test_invalid_discount_is_rejected() -> None:
    from algorithms import value_iteration

    with pytest.raises(ValueError):
        value_iteration(GridWorld(), gamma=1.0)

