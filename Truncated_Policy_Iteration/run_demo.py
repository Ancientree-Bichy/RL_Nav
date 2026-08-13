"""Run and visualize all dynamic-programming Grid World solvers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from algorithms import (
    AlgorithmResult,
    policy_iteration,
    truncated_policy_iteration,
    value_iteration,
)
from grid_world import GridWorld
from visualization import (
    save_algorithm_comparison,
    save_convergence_plot,
    save_value_animation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare value iteration, policy iteration, and truncated PI."
    )
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--theta", type=float, default=1e-8)
    parser.add_argument(
        "--truncated-sweeps",
        type=int,
        default=3,
        help="Policy-evaluation sweeps per truncated-PI improvement (m).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip the animated GIF for a faster run.",
    )
    return parser.parse_args()


def run_all(
    env: GridWorld, gamma: float, theta: float, truncated_sweeps: int
) -> list[AlgorithmResult]:
    return [
        value_iteration(env, gamma=gamma, theta=theta),
        policy_iteration(env, gamma=gamma, theta=theta),
        truncated_policy_iteration(
            env,
            evaluation_sweeps_per_improvement=truncated_sweeps,
            gamma=gamma,
            theta=theta,
        ),
    ]


def _summary(result: AlgorithmResult, env: GridWorld) -> dict[str, object]:
    return {
        "algorithm": result.name,
        "outer_steps": result.improvement_steps,
        "evaluation_sweeps": result.evaluation_sweeps,
        "state_backups": result.state_backups,
        "path_length": result.path_length,
        "reaches_goal": env.validate_path(result.path),
        "start_value": float(result.values[env.start]),
        "path": [list(state) for state in result.path],
    }


def main() -> None:
    args = parse_args()
    env = GridWorld()
    results = run_all(env, args.gamma, args.theta, args.truncated_sweeps)

    for result in results:
        if not env.validate_path(result.path):
            raise RuntimeError(f"{result.name} did not recover a valid goal path")

    optimal_values = results[0].values
    for result in results[1:]:
        max_difference = max(
            abs(result.values[state] - optimal_values[state])
            for state in env.nonterminal_states
        )
        if max_difference > max(1e-6, args.theta * 100):
            raise RuntimeError(
                f"{result.name} differs from value iteration by {max_difference:.3g}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "algorithm_comparison.png"
    convergence_path = args.output_dir / "convergence.png"
    summary_path = args.output_dir / "results.json"

    save_algorithm_comparison(env, results, comparison_path)
    save_convergence_plot(env, results, optimal_values, convergence_path)
    summary = {
        "grid": {
            "height": env.height,
            "width": env.width,
            "start": list(env.start),
            "goal": list(env.goal),
            "obstacle_count": len(env.obstacles),
            "gamma": args.gamma,
            "theta": args.theta,
        },
        "results": [_summary(result, env) for result in results],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not args.no_animation:
        animation_path = args.output_dir / "value_evolution.gif"
        save_value_animation(env, results, animation_path, gamma=args.gamma)

    print(f"{'Algorithm':<36} {'Outer':>7} {'Sweeps':>8} {'Backups':>9} {'Path':>6}")
    print("-" * 72)
    for result in results:
        print(
            f"{result.name:<36} "
            f"{result.improvement_steps:>7} "
            f"{result.evaluation_sweeps:>8} "
            f"{result.state_backups:>9} "
            f"{result.path_length:>6}"
        )
    print(f"\nSaved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

