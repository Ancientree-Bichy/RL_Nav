"""A small deterministic Grid World expressed as a finite MDP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

State = tuple[int, int]

# The action order is also the deterministic tie-breaking order.
ACTION_DELTAS: tuple[State, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES: tuple[str, ...] = ("up", "right", "down", "left")
ACTION_ARROWS: tuple[str, ...] = ("↑", "→", "↓", "←")


def default_obstacles() -> frozenset[State]:
    """Return three walls with gaps that force a visible zig-zag route."""

    return frozenset(
        {
            # Lower wall; its useful gap is at column 1.
            (7, 0),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 8),
            (7, 9),
            # Middle wall; cross it at column 7.
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 8),
            (4, 9),
            # Upper wall; cross it at column 4.
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
        }
    )


@dataclass(frozen=True)
class GridWorld:
    """A deterministic 4-neighbour Grid World.

    Every ordinary move costs ``step_reward``. Entering the terminal goal gives
    ``goal_reward`` and no further reward. Hitting a wall, obstacle, or boundary
    leaves the agent in place and still incurs the step cost.
    """

    height: int = 10
    width: int = 10
    start: State = (9, 0)
    goal: State = (0, 9)
    obstacles: frozenset[State] = field(default_factory=default_obstacles)
    step_reward: float = -1.0
    goal_reward: float = 0.0

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if not self._inside(self.start) or not self._inside(self.goal):
            raise ValueError("start and goal must be inside the grid")
        if self.start == self.goal:
            raise ValueError("start and goal must be different")
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("start and goal cannot be obstacles")
        if any(not self._inside(state) for state in self.obstacles):
            raise ValueError("all obstacles must be inside the grid")

    @property
    def n_actions(self) -> int:
        return len(ACTION_DELTAS)

    @property
    def states(self) -> tuple[State, ...]:
        return tuple(
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if (row, col) not in self.obstacles
        )

    @property
    def nonterminal_states(self) -> tuple[State, ...]:
        return tuple(state for state in self.states if state != self.goal)

    def _inside(self, state: State) -> bool:
        row, col = state
        return 0 <= row < self.height and 0 <= col < self.width

    def is_navigable(self, state: State) -> bool:
        return self._inside(state) and state not in self.obstacles

    def transition(self, state: State, action: int) -> tuple[State, float]:
        """Return the deterministic next state and reward."""

        if not self.is_navigable(state):
            raise ValueError(f"invalid state: {state}")
        if not 0 <= action < self.n_actions:
            raise ValueError(f"invalid action: {action}")
        if state == self.goal:
            return self.goal, 0.0

        dr, dc = ACTION_DELTAS[action]
        candidate = (state[0] + dr, state[1] + dc)
        next_state = candidate if self.is_navigable(candidate) else state
        reward = self.goal_reward if next_state == self.goal else self.step_reward
        return next_state, reward

    def action_values(
        self, state: State, values: np.ndarray, gamma: float
    ) -> np.ndarray:
        """Compute Q(s, ·) from a value table."""

        if state == self.goal:
            return np.zeros(self.n_actions, dtype=float)
        q_values = np.empty(self.n_actions, dtype=float)
        for action in range(self.n_actions):
            next_state, reward = self.transition(state, action)
            q_values[action] = reward + gamma * values[next_state]
        return q_values

    def empty_values(self) -> np.ndarray:
        """Create a zero value table with obstacles marked as NaN."""

        values = np.zeros((self.height, self.width), dtype=float)
        for obstacle in self.obstacles:
            values[obstacle] = np.nan
        return values

    def uniform_policy(self) -> np.ndarray:
        """Create a uniform random policy over all four actions."""

        policy = np.zeros((self.height, self.width, self.n_actions), dtype=float)
        probability = 1.0 / self.n_actions
        for state in self.nonterminal_states:
            policy[state] = probability
        return policy

    def greedy_policy(self, values: np.ndarray, gamma: float) -> np.ndarray:
        """Return a deterministic policy greedy with respect to ``values``."""

        policy = np.zeros((self.height, self.width, self.n_actions), dtype=float)
        for state in self.nonterminal_states:
            best_action = int(np.argmax(self.action_values(state, values, gamma)))
            policy[state][best_action] = 1.0
        return policy

    def extract_path(
        self, policy: np.ndarray, max_steps: int | None = None
    ) -> list[State]:
        """Follow a deterministic policy from start until goal or a loop."""

        if policy.shape != (self.height, self.width, self.n_actions):
            raise ValueError("policy has an incompatible shape")

        max_steps = max_steps or self.height * self.width * 4
        path = [self.start]
        visited = {self.start}
        state = self.start

        for _ in range(max_steps):
            if state == self.goal:
                break
            action = int(np.argmax(policy[state]))
            next_state, _ = self.transition(state, action)
            path.append(next_state)
            if next_state == self.goal:
                break
            if next_state in visited:
                break
            visited.add(next_state)
            state = next_state
        return path

    def validate_path(self, path: Iterable[State]) -> bool:
        """Return whether a path is legal and reaches the goal."""

        states = list(path)
        if not states or states[0] != self.start or states[-1] != self.goal:
            return False
        for state, next_state in zip(states, states[1:]):
            if not self.is_navigable(next_state):
                return False
            distance = abs(state[0] - next_state[0]) + abs(state[1] - next_state[1])
            if distance != 1:
                return False
        return True
