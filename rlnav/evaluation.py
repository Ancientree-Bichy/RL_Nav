from __future__ import annotations

import torch

from rlnav.algorithms.base import RLAgent
from rlnav.envs.gridworld import GridWorld


@torch.no_grad()
def rollout(
    env: GridWorld,
    agent: RLAgent,
    max_steps: int = 200,
    render: bool = False,
) -> tuple[float, int, bool]:
    result = rollout_with_stats(env=env, agent=agent, max_steps=max_steps, render=render)
    return result["return"], result["steps"], result["success"]


@torch.no_grad()
def rollout_with_stats(
    env: GridWorld,
    agent: RLAgent,
    max_steps: int = 200,
    render: bool = False,
) -> dict:
    state = env.reset()
    done = False
    total_return = 0.0
    steps = 0
    collisions = 0
    obstacle_hits = 0
    wall_hits = 0

    if render:
        print("\n[Rollout] initial:\n" + env.render_ascii() + "\n")

    while not done and steps < max_steps:
        action = agent.act(state, explore=False)
        state, reward, done, info = env.step(action)
        total_return += reward
        steps += 1
        if info.get("collision", False):
            collisions += 1
            ctype = info.get("collision_type", "")
            if ctype == "obstacle":
                obstacle_hits += 1
            elif ctype == "wall":
                wall_hits += 1

        if render:
            print(f"step={steps:3d}, action={action}, reward={reward:+.2f}")
            print(env.render_ascii() + "\n")

    success = env.at_goal
    return {
        "return": float(total_return),
        "steps": int(steps),
        "success": bool(success),
        "collisions": int(collisions),
        "wall_hits": int(wall_hits),
        "obstacle_hits": int(obstacle_hits),
    }
