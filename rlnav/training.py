from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from rlnav.algorithms.base import RLAgent
from rlnav.envs.gridworld import GridWorld
from rlnav.infra.checkpoint import save_checkpoint
from rlnav.infra.logger import TrainingLogger


def train(
    env: GridWorld,
    agent: RLAgent,
    episodes: int,
    logger: TrainingLogger,
    algo_name: str,
    env_config: dict[str, Any],
    train_config: dict[str, Any],
    checkpoint_dir: str | Path,
    log_every: int = 50,
    save_every: int = 200,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_return = float("-inf")

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        agent.start_episode()
        step_metrics: list[dict[str, float]] = []

        while not done:
            action = agent.act(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            metrics = agent.observe(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            if metrics:
                step_metrics.append(metrics)

            state = next_state
            ep_return += reward
            steps += 1

        ep_end_metrics = agent.end_episode()
        merged_metrics = _merge_metrics(step_metrics)
        merged_metrics.update(ep_end_metrics)
        row = logger.log_episode(
            episode=ep,
            episode_return=ep_return,
            steps=steps,
            success=env.at_goal,
            extra_metrics=merged_metrics,
        )

        if ep % log_every == 0:
            avg_ret_key = f"avg_return_last_{logger.window}"
            avg_suc_key = f"avg_success_last_{logger.window}"
            msg = (
                f"[{algo_name.upper():4s}] ep={ep:4d} "
                f"return={ep_return:+.3f} "
                f"{avg_ret_key}={row[avg_ret_key]:+.3f} "
                f"{avg_suc_key}={row[avg_suc_key]:.3f}"
            )
            if "loss" in row:
                msg += f" loss={row['loss']:.4f}"
            if "epsilon" in row:
                msg += f" eps={row['epsilon']:.3f}"
            print(msg)

        payload = {
            "algo": algo_name,
            "episode": ep,
            "best_return": best_return,
            "env_config": env_config,
            "train_config": train_config,
            "agent_state": agent.state_dict(),
        }

        if ep_return > best_return:
            best_return = ep_return
            payload["best_return"] = best_return
            save_checkpoint(checkpoint_dir / "best.pt", payload)

        if ep % save_every == 0:
            save_checkpoint(checkpoint_dir / "last.pt", payload)

    final_payload = {
        "algo": algo_name,
        "episode": episodes,
        "best_return": best_return,
        "env_config": env_config,
        "train_config": train_config,
        "agent_state": agent.state_dict(),
    }
    save_checkpoint(checkpoint_dir / "last.pt", final_payload)

    return {
        "best_return": best_return,
        "checkpoint_dir": str(checkpoint_dir),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
    }


def _merge_metrics(step_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not step_metrics:
        return {}
    all_keys = sorted(set().union(*[m.keys() for m in step_metrics]))
    merged: dict[str, float] = {}
    for key in all_keys:
        values = [m[key] for m in step_metrics if key in m]
        if values:
            merged[key] = float(mean(values))
    return merged
