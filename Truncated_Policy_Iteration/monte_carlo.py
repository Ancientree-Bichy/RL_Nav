"""First-visit Monte Carlo control for the Grid World.

Unlike the dynamic-programming algorithms in ``algorithms.py``, training here
does not use ``GridWorld.action_values`` or a transition table. The agent only
observes sampled ``(state, action, reward, next_state)`` transitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid_world import GridWorld, State


@dataclass
class MonteCarloResult:
    """Outputs and diagnostics from first-visit Monte Carlo control."""

    name: str
    q_values: np.ndarray
    values: np.ndarray
    policy: np.ndarray
    path: list[State]
    visit_counts: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    successes: np.ndarray
    epsilons: np.ndarray
    seed: int

    @property
    def episodes(self) -> int:
        return len(self.episode_returns)

    @property
    def path_length(self) -> int:
        return max(0, len(self.path) - 1)

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.successes))


def _check_parameters(
    episodes: int,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    max_steps: int,
    start_state_probability: float,
) -> None:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
        raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0.0 <= start_state_probability <= 1.0:
        raise ValueError("start_state_probability must be in [0, 1]")


def _epsilon_for_episode(
    episode: int, episodes: int, epsilon_start: float, epsilon_end: float
) -> float:
    """Linearly anneal epsilon, including both requested endpoints."""

    if episodes == 1:
        return epsilon_end
    fraction = episode / (episodes - 1)
    return epsilon_start + fraction * (epsilon_end - epsilon_start)


def _sample_epsilon_greedy_action(
    q_values: np.ndarray,
    state: State,
    epsilon: float,
    n_actions: int,
    rng: np.random.Generator,
) -> int:
    """Sample an epsilon-greedy action with random tie-breaking."""

    if rng.random() < epsilon:
        return int(rng.integers(n_actions))

    state_q = q_values[state]
    best_actions = np.flatnonzero(
        np.isclose(state_q, np.max(state_q), rtol=1e-10, atol=1e-12)
    )
    return int(rng.choice(best_actions))


def _generate_episode(
    env: GridWorld,
    q_values: np.ndarray,
    epsilon: float,
    max_steps: int,
    start_state_probability: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[State, int, float]], bool]:
    """Interact with the environment to sample one finite episode."""

    if rng.random() < start_state_probability:
        state = env.start
    else:
        state = env.nonterminal_states[
            int(rng.integers(len(env.nonterminal_states)))
        ]

    trajectory: list[tuple[State, int, float]] = []
    for _ in range(max_steps):
        action = _sample_epsilon_greedy_action(
            q_values, state, epsilon, env.n_actions, rng
        )
        next_state, reward = env.transition(state, action)
        trajectory.append((state, action, reward))
        state = next_state
        if state == env.goal:
            return trajectory, True
    return trajectory, False


def _first_visit_update(
    trajectory: list[tuple[State, int, float]],
    q_values: np.ndarray,
    visit_counts: np.ndarray,
    gamma: float,
) -> float:
    """Apply sample-average first-visit MC updates and return G₀."""

    returns = np.empty(len(trajectory), dtype=float)
    discounted_return = 0.0
    for index in range(len(trajectory) - 1, -1, -1):
        _, _, reward = trajectory[index]
        discounted_return = reward + gamma * discounted_return
        returns[index] = discounted_return

    seen: set[tuple[State, int]] = set()
    for index, (state, action, _) in enumerate(trajectory):
        state_action = (state, action)
        if state_action in seen:
            continue
        seen.add(state_action)

        visit_counts[state][action] += 1
        count = visit_counts[state][action]
        # Incremental sample mean: Q_n = Q_{n-1} + (G - Q_{n-1}) / n.
        q_values[state][action] += (
            returns[index] - q_values[state][action]
        ) / count

    return float(returns[0]) if len(returns) else 0.0


def _greedy_policy_from_q(env: GridWorld, q_values: np.ndarray) -> np.ndarray:
    """Create a deterministic policy directly from learned action values."""

    policy = np.zeros((env.height, env.width, env.n_actions), dtype=float)
    for state in env.nonterminal_states:
        action = int(np.argmax(q_values[state]))
        policy[state][action] = 1.0
    return policy


def _state_values_from_q(env: GridWorld, q_values: np.ndarray) -> np.ndarray:
    values = env.empty_values()
    for state in env.nonterminal_states:
        values[state] = float(np.max(q_values[state]))
    values[env.goal] = 0.0
    return values


def first_visit_mc_control(
    env: GridWorld,
    episodes: int = 30_000,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,
    max_steps: int = 400,
    start_state_probability: float = 0.5,
    seed: int = 7,
) -> MonteCarloResult:
    """Learn an epsilon-greedy policy with first-visit Monte Carlo control.

    Half of the default episodes begin at the configured start, while the other
    half use exploring starts sampled across the grid. This retains the original
    navigation task while giving every state-action pair a chance to be learned.
    """

    _check_parameters(
        episodes,
        gamma,
        epsilon_start,
        epsilon_end,
        max_steps,
        start_state_probability,
    )
    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.height, env.width, env.n_actions), dtype=float)
    visit_counts = np.zeros_like(q_values, dtype=np.int64)
    episode_returns = np.empty(episodes, dtype=float)
    episode_lengths = np.empty(episodes, dtype=np.int64)
    successes = np.zeros(episodes, dtype=bool)
    epsilons = np.empty(episodes, dtype=float)

    for episode in range(episodes):
        epsilon = _epsilon_for_episode(
            episode, episodes, epsilon_start, epsilon_end
        )
        trajectory, success = _generate_episode(
            env,
            q_values,
            epsilon,
            max_steps,
            start_state_probability,
            rng,
        )
        episode_returns[episode] = _first_visit_update(
            trajectory, q_values, visit_counts, gamma
        )
        episode_lengths[episode] = len(trajectory)
        successes[episode] = success
        epsilons[episode] = epsilon

    policy = _greedy_policy_from_q(env, q_values)
    values = _state_values_from_q(env, q_values)
    return MonteCarloResult(
        name="First-Visit Monte Carlo Control",
        q_values=q_values,
        values=values,
        policy=policy,
        path=env.extract_path(policy),
        visit_counts=visit_counts,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        successes=successes,
        epsilons=epsilons,
        seed=seed,
    )

