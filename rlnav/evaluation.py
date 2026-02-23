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
    state = env.reset()
    done = False
    total_return = 0.0
    steps = 0

    if render:
        print("\n[Rollout] initial:\n" + env.render_ascii() + "\n")

    while not done and steps < max_steps:
        action = agent.act(state, explore=False)
        state, reward, done, _ = env.step(action)
        total_return += reward
        steps += 1

        if render:
            print(f"step={steps:3d}, action={action}, reward={reward:+.2f}")
            print(env.render_ascii() + "\n")

    success = env.at_goal
    return total_return, steps, success
