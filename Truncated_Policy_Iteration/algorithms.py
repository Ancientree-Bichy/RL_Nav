"""Dynamic-programming algorithms for the finite Grid World MDP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid_world import GridWorld, State


@dataclass
class AlgorithmResult:
    """Common result format used by all three algorithms."""

    name: str
    values: np.ndarray
    policy: np.ndarray
    path: list[State]
    improvement_steps: int
    evaluation_sweeps: int
    state_backups: int
    deltas: list[float]
    value_history: list[np.ndarray]
    policy_changes: list[int]

    @property
    def path_length(self) -> int:
        return max(0, len(self.path) - 1)


def _check_parameters(gamma: float, theta: float, max_iterations: int) -> None:
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")


def _policy_evaluation_sweep(
    env: GridWorld,
    policy: np.ndarray,
    values: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, float]:
    """Perform one synchronous Bellman expectation backup over every state."""

    new_values = values.copy()
    delta = 0.0
    for state in env.nonterminal_states:
        q_values = env.action_values(state, values, gamma)
        updated = float(np.dot(policy[state], q_values))
        delta = max(delta, abs(updated - values[state]))
        new_values[state] = updated
    new_values[env.goal] = 0.0
    return new_values, delta


def bellman_optimality_residual(
    env: GridWorld, values: np.ndarray, gamma: float
) -> float:
    """Return ||T*V - V||∞ over nonterminal states."""

    return max(
        abs(float(np.max(env.action_values(state, values, gamma))) - values[state])
        for state in env.nonterminal_states
    )


def _policy_difference(
    env: GridWorld, old_policy: np.ndarray, new_policy: np.ndarray
) -> int:
    return sum(
        int(not np.allclose(old_policy[state], new_policy[state]))
        for state in env.nonterminal_states
    )


def value_iteration(
    env: GridWorld,
    gamma: float = 0.95,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
) -> AlgorithmResult:
    """Solve the MDP with repeated Bellman optimality backups."""

    _check_parameters(gamma, theta, max_iterations)
    values = env.empty_values()
    history = [values.copy()]
    deltas: list[float] = []

    for _ in range(max_iterations):
        new_values = values.copy()
        delta = 0.0
        for state in env.nonterminal_states:
            updated = float(np.max(env.action_values(state, values, gamma)))
            delta = max(delta, abs(updated - values[state]))
            new_values[state] = updated
        new_values[env.goal] = 0.0
        values = new_values
        deltas.append(delta)
        history.append(values.copy())
        if delta < theta:
            break
    else:
        raise RuntimeError("value iteration did not converge")

    policy = env.greedy_policy(values, gamma)
    return AlgorithmResult(
        name="Value Iteration",
        values=values,
        policy=policy,
        path=env.extract_path(policy),
        improvement_steps=len(deltas),
        evaluation_sweeps=len(deltas),
        state_backups=len(deltas) * len(env.nonterminal_states),
        deltas=deltas,
        value_history=history,
        policy_changes=[],
    )


def policy_iteration(
    env: GridWorld,
    gamma: float = 0.95,
    theta: float = 1e-8,
    max_iterations: int = 1_000,
    max_evaluation_sweeps: int = 100_000,
) -> AlgorithmResult:
    """Alternate exact policy evaluation and greedy policy improvement."""

    _check_parameters(gamma, theta, max_iterations)
    policy = env.uniform_policy()
    values = env.empty_values()
    history = [values.copy()]
    deltas: list[float] = []
    policy_changes: list[int] = []
    total_sweeps = 0

    for improvement_step in range(1, max_iterations + 1):
        # Evaluate the current policy to the requested tolerance.
        for _ in range(max_evaluation_sweeps):
            values, delta = _policy_evaluation_sweep(env, policy, values, gamma)
            total_sweeps += 1
            deltas.append(delta)
            history.append(values.copy())
            if delta < theta:
                break
        else:
            raise RuntimeError("policy evaluation did not converge")

        improved_policy = env.greedy_policy(values, gamma)
        changed = _policy_difference(env, policy, improved_policy)
        policy_changes.append(changed)
        policy = improved_policy
        if changed == 0:
            break
    else:
        raise RuntimeError("policy iteration did not converge")

    return AlgorithmResult(
        name="Policy Iteration",
        values=values,
        policy=policy,
        path=env.extract_path(policy),
        improvement_steps=improvement_step,
        evaluation_sweeps=total_sweeps,
        state_backups=total_sweeps * len(env.nonterminal_states),
        deltas=deltas,
        value_history=history,
        policy_changes=policy_changes,
    )


def truncated_policy_iteration(
    env: GridWorld,
    evaluation_sweeps_per_improvement: int = 3,
    gamma: float = 0.95,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
) -> AlgorithmResult:
    """Policy iteration with only a fixed number of evaluation sweeps.

    This is also called modified policy iteration. It interpolates between
    value iteration (very short evaluation) and full policy iteration.
    """

    _check_parameters(gamma, theta, max_iterations)
    if evaluation_sweeps_per_improvement <= 0:
        raise ValueError("evaluation_sweeps_per_improvement must be positive")

    policy = env.uniform_policy()
    values = env.empty_values()
    history = [values.copy()]
    deltas: list[float] = []
    policy_changes: list[int] = []
    total_sweeps = 0

    for improvement_step in range(1, max_iterations + 1):
        for _ in range(evaluation_sweeps_per_improvement):
            values, delta = _policy_evaluation_sweep(env, policy, values, gamma)
            total_sweeps += 1
            deltas.append(delta)
            history.append(values.copy())

        improved_policy = env.greedy_policy(values, gamma)
        changed = _policy_difference(env, policy, improved_policy)
        policy_changes.append(changed)
        policy = improved_policy

        # A stable greedy policy alone is not enough when V is only partially
        # evaluated. The optimality residual also has to be small.
        residual = bellman_optimality_residual(env, values, gamma)
        if changed == 0 and residual < theta:
            break
    else:
        raise RuntimeError("truncated policy iteration did not converge")

    return AlgorithmResult(
        name=f"Truncated Policy Iteration (m={evaluation_sweeps_per_improvement})",
        values=values,
        policy=policy,
        path=env.extract_path(policy),
        improvement_steps=improvement_step,
        evaluation_sweeps=total_sweeps,
        state_backups=total_sweeps * len(env.nonterminal_states),
        deltas=deltas,
        value_history=history,
        policy_changes=policy_changes,
    )
