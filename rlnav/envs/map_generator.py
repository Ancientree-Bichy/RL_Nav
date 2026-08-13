from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

from .gridworld import GridWorld, GridWorldConfig


@dataclass
class RandomMapConfig:
    height: int
    width: int
    start: tuple[int, int]
    goal: tuple[int, int]
    obstacle_prob: float = 0.2
    max_steps: int = 200
    ensure_path: bool = True
    max_attempts: int = 200


def build_random_gridworld(cfg: RandomMapConfig, rng: random.Random) -> GridWorld:
    for _ in range(max(1, cfg.max_attempts)):
        obstacles = _sample_obstacles(cfg=cfg, rng=rng)
        if not cfg.ensure_path or _has_path(
            height=cfg.height,
            width=cfg.width,
            start=cfg.start,
            goal=cfg.goal,
            obstacles=set(obstacles),
        ):
            env_cfg = GridWorldConfig(
                height=cfg.height,
                width=cfg.width,
                start=cfg.start,
                goal=cfg.goal,
                max_steps=cfg.max_steps,
                obstacles=obstacles,
            )
            return GridWorld.from_config(env_cfg)
    raise RuntimeError(
        "Failed to generate a random map with a valid path. "
        "Try lower --map-obstacle-prob or increase --random-map-max-attempts."
    )


def _sample_obstacles(cfg: RandomMapConfig, rng: random.Random) -> list[tuple[int, int]]:
    p = min(max(cfg.obstacle_prob, 0.0), 0.95)
    obstacles: list[tuple[int, int]] = []
    for r in range(cfg.height):
        for c in range(cfg.width):
            pos = (r, c)
            if pos == cfg.start or pos == cfg.goal:
                continue
            if rng.random() < p:
                obstacles.append(pos)
    return obstacles


def _has_path(
    height: int,
    width: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: set[tuple[int, int]],
) -> bool:
    if start == goal:
        return True
    if start in obstacles or goal in obstacles:
        return False

    q = deque([start])
    visited = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if nxt in obstacles or nxt in visited:
                continue
            if nxt == goal:
                return True
            visited.add(nxt)
            q.append(nxt)
    return False
