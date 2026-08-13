"""Train and visualize first-visit Monte Carlo control on the Grid World."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from grid_world import GridWorld
from monte_carlo import first_visit_mc_control
from visualization import save_monte_carlo_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="First-visit Monte Carlo control Grid World demo."
    )
    parser.add_argument("--episodes", type=int, default=30_000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument(
        "--start-probability",
        type=float,
        default=0.5,
        help="Probability that an episode begins at the configured start.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = GridWorld()
    result = first_visit_mc_control(
        env,
        episodes=args.episodes,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        max_steps=args.max_steps,
        start_state_probability=args.start_probability,
        seed=args.seed,
    )

    if not env.validate_path(result.path):
        raise RuntimeError(
            "The learned greedy policy did not reach the goal. "
            "Try more episodes or a higher final epsilon."
        )

    visited_pairs = sum(
        int(result.visit_counts[state][action] > 0)
        for state in env.nonterminal_states
        for action in range(env.n_actions)
    )
    total_pairs = len(env.nonterminal_states) * env.n_actions
    recent_window = min(1_000, result.episodes)
    recent_success = float(np.mean(result.successes[-recent_window:]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "monte_carlo_summary.png"
    result_path = args.output_dir / "monte_carlo_results.json"
    save_monte_carlo_summary(env, result, figure_path)
    result_path.write_text(
        json.dumps(
            {
                "algorithm": result.name,
                "episodes": result.episodes,
                "seed": result.seed,
                "gamma": args.gamma,
                "epsilon_start": args.epsilon_start,
                "epsilon_end": args.epsilon_end,
                "max_steps": args.max_steps,
                "start_state_probability": args.start_probability,
                "overall_episode_success_rate": result.success_rate,
                "recent_success_rate": recent_success,
                "visited_state_action_pairs": visited_pairs,
                "total_state_action_pairs": total_pairs,
                "coverage": visited_pairs / total_pairs,
                "path_length": result.path_length,
                "reaches_goal": env.validate_path(result.path),
                "start_value": float(result.values[env.start]),
                "path": [list(state) for state in result.path],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Algorithm:               {result.name}")
    print(f"Episodes:                {result.episodes}")
    print(f"Overall success rate:    {result.success_rate:.1%}")
    print(f"Last {recent_window} success rate: {recent_success:.1%}")
    print(f"State-action coverage:   {visited_pairs}/{total_pairs}")
    print(f"Learned path length:     {result.path_length}")
    print(f"Saved results to:        {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

