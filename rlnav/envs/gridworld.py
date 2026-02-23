from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_OBSTACLES = [
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 3),
    (3, 3),
    (4, 3),
    (5, 1),
    (5, 2),
    (5, 3),
    (5, 4),
    (6, 4),
    (6, 5),
]


@dataclass
class GridWorldConfig:
    height: int = 8
    width: int = 8
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (7, 7)
    max_steps: int = 200
    obstacles: list[tuple[int, int]] = field(default_factory=lambda: list(DEFAULT_OBSTACLES))

    def to_dict(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "start": list(self.start),
            "goal": list(self.goal),
            "max_steps": self.max_steps,
            "obstacles": [list(x) for x in self.obstacles],
        }


def default_grid_config(max_steps: int = 200) -> GridWorldConfig:
    return GridWorldConfig(max_steps=max_steps)


class GridWorld:
    """
    Deterministic 2D grid navigation:
      - State: one-hot over H*W
      - Actions: 0=up,1=down,2=left,3=right
      - Reward: step=-0.01, invalid move=-0.05 extra, goal=+1.0
    """

    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        obstacles: list[tuple[int, int]] | set[tuple[int, int]] | None = None,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (7, 7),
        max_steps: int = 200,
    ):
        self.H = height
        self.W = width
        self.start = start
        self.goal = goal
        self.max_steps = max_steps

        self.obstacles = set(obstacles or [])
        self.obstacles.discard(self.start)
        self.obstacles.discard(self.goal)

        self.agent_r = self.start[0]
        self.agent_c = self.start[1]
        self.t = 0

    @classmethod
    def from_config(cls, cfg: GridWorldConfig) -> "GridWorld":
        return cls(
            height=cfg.height,
            width=cfg.width,
            obstacles=cfg.obstacles,
            start=cfg.start,
            goal=cfg.goal,
            max_steps=cfg.max_steps,
        )

    @property
    def n_states(self) -> int:
        return self.H * self.W

    @property
    def n_actions(self) -> int:
        return 4

    @property
    def at_goal(self) -> bool:
        return (self.agent_r, self.agent_c) == self.goal

    def _idx(self, r: int, c: int) -> int:
        return r * self.W + c

    def obs(self) -> np.ndarray:
        x = np.zeros(self.n_states, dtype=np.float32)
        x[self._idx(self.agent_r, self.agent_c)] = 1.0
        return x

    def reset(self) -> np.ndarray:
        self.agent_r, self.agent_c = self.start
        self.t = 0
        return self.obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.t += 1
        r0, c0 = self.agent_r, self.agent_c

        drdc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = drdc[action]
        r1, c1 = r0 + dr, c0 + dc

        reward = -0.01
        done = False

        if not (0 <= r1 < self.H and 0 <= c1 < self.W):
            r1, c1 = r0, c0
            reward += -0.05
        elif (r1, c1) in self.obstacles:
            r1, c1 = r0, c0
            reward += -0.05

        self.agent_r, self.agent_c = r1, c1

        if (r1, c1) == self.goal:
            reward = 1.0
            done = True

        if self.t >= self.max_steps:
            done = True

        return self.obs(), float(reward), done, {}

    def render_ascii(self) -> str:
        grid = []
        for r in range(self.H):
            row = []
            for c in range(self.W):
                if (r, c) == (self.agent_r, self.agent_c):
                    row.append("A")
                elif (r, c) == self.start:
                    row.append("S")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) in self.obstacles:
                    row.append("#")
                else:
                    row.append(".")
            grid.append(" ".join(row))
        return "\n".join(grid)
