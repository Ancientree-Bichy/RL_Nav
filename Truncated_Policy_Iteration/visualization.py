"""Matplotlib visualizations shared by all Grid World algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

# A non-interactive backend makes scripts and tests work over SSH and in CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

from algorithms import AlgorithmResult
from grid_world import ACTION_ARROWS, GridWorld
from monte_carlo import MonteCarloResult

ALGORITHM_COLORS = ("#0072B2", "#D55E00", "#009E73")


class _FullFramePillowWriter(animation.PillowWriter):
    """Pillow writer that clears each GIF frame before drawing the next one."""

    def finish(self) -> None:
        # constrained_layout can shift axes by a few pixels as titles change.
        # disposal=2 prevents old tick labels from surviving in those pixels.
        self._frames[0].save(
            self.outfile,
            save_all=True,
            append_images=self._frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
            disposal=2,
        )


def _draw_grid(
    ax: Axes,
    env: GridWorld,
    values: np.ndarray,
    policy: np.ndarray,
    path: Sequence[tuple[int, int]] | None,
    title: str,
    norm: colors.Normalize,
    show_numbers: bool = True,
):
    """Draw values, greedy actions, obstacles, and a path on one axis."""

    masked_values = np.ma.masked_invalid(values)
    image = ax.imshow(masked_values, cmap="viridis", norm=norm, origin="upper")

    for row, col in env.obstacles:
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1,
                1,
                facecolor="#3B3B3B",
                edgecolor="#111111",
                linewidth=0.8,
                zorder=3,
            )
        )

    if path and len(path) > 1:
        path_rows = [state[0] for state in path]
        path_cols = [state[1] for state in path]
        ax.plot(
            path_cols,
            path_rows,
            color="#F5C242",
            linewidth=3.2,
            marker="o",
            markersize=3,
            label=f"path ({len(path) - 1} steps)",
            zorder=5,
        )

    for state in env.nonterminal_states:
        row, col = state
        action = int(np.argmax(policy[state]))
        value_text = f"{values[state]:.1f}\n" if show_numbers else ""
        ax.text(
            col,
            row,
            value_text + ACTION_ARROWS[action],
            ha="center",
            va="center",
            fontsize=6.3 if show_numbers else 8,
            color="white",
            fontweight="bold",
            zorder=6,
        )

    start_row, start_col = env.start
    goal_row, goal_col = env.goal
    ax.scatter(
        [start_col],
        [start_row],
        s=110,
        marker="o",
        facecolor="#56B4E9",
        edgecolor="white",
        linewidth=1.5,
        zorder=7,
        label="start",
    )
    ax.scatter(
        [goal_col],
        [goal_row],
        s=180,
        marker="*",
        facecolor="#E69F00",
        edgecolor="white",
        linewidth=1.2,
        zorder=7,
        label="goal",
    )
    ax.text(
        start_col,
        start_row,
        "S",
        ha="center",
        va="center",
        fontsize=6,
        fontweight="bold",
        color="#111111",
        zorder=8,
    )
    ax.text(
        goal_col,
        goal_row,
        "G",
        ha="center",
        va="center",
        fontsize=6,
        fontweight="bold",
        color="#111111",
        zorder=8,
    )

    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title(title, fontsize=11)
    return image


def save_algorithm_comparison(
    env: GridWorld, results: Sequence[AlgorithmResult], output_path: Path
) -> None:
    """Save one side-by-side value/policy/path panel per algorithm."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite_values = np.concatenate(
        [result.values[np.isfinite(result.values)] for result in results]
    )
    norm = colors.Normalize(vmin=float(finite_values.min()), vmax=0.0)
    figure, axes = plt.subplots(1, len(results), figsize=(19, 6.8), constrained_layout=True)
    axes = np.atleast_1d(axes)

    image = None
    for ax, result in zip(axes, results):
        subtitle = (
            f"{result.name}\n"
            f"{result.improvement_steps} outer steps · "
            f"{result.evaluation_sweeps} sweeps · "
            f"path {result.path_length}"
        )
        image = _draw_grid(
            ax,
            env,
            result.values,
            result.policy,
            result.path,
            subtitle,
            norm,
        )

    assert image is not None
    colorbar = figure.colorbar(image, ax=axes.tolist(), shrink=0.78, pad=0.02)
    colorbar.set_label("State value V(s); 0 is the terminal goal")
    figure.suptitle(
        "10×10 Grid World: values, greedy policies, and recovered paths",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_convergence_plot(
    env: GridWorld,
    results: Sequence[AlgorithmResult],
    reference_values: np.ndarray,
    output_path: Path,
) -> None:
    """Compare value-table error against cumulative state backups."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    states_per_sweep = len(env.nonterminal_states)

    for result, color in zip(results, ALGORITHM_COLORS):
        errors = []
        for values in result.value_history:
            differences = [
                abs(values[state] - reference_values[state])
                for state in env.nonterminal_states
            ]
            errors.append(max(max(differences), 1e-12))
        backups = np.arange(len(errors)) * states_per_sweep
        ax.semilogy(
            backups,
            errors,
            color=color,
            linewidth=2,
            label=result.name,
        )
        ax.scatter([backups[-1]], [errors[-1]], color=color, s=25, zorder=3)

    ax.set_xlabel("Cumulative state backups")
    ax.set_ylabel(r"$\|V_k - V^*\|_\infty$")
    ax.set_title("Convergence toward the optimal value function")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _history_index(history_length: int, frame: int, frame_count: int) -> int:
    if frame_count <= 1:
        return history_length - 1
    return round(frame * (history_length - 1) / (frame_count - 1))


def save_value_animation(
    env: GridWorld,
    results: Sequence[AlgorithmResult],
    output_path: Path,
    gamma: float,
    max_frames: int = 36,
) -> None:
    """Animate the evolution of values and each intermediate greedy route."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite_values = np.concatenate(
        [
            values[np.isfinite(values)]
            for result in results
            for values in result.value_history
        ]
    )
    norm = colors.Normalize(vmin=float(finite_values.min()), vmax=0.0)
    longest_history = max(len(result.value_history) for result in results)
    frame_count = min(max_frames, longest_history)
    figure, axes = plt.subplots(1, len(results), figsize=(16.5, 5.5), constrained_layout=True)
    axes = np.atleast_1d(axes)

    def update(frame: int):
        artists = []
        for ax, result in zip(axes, results):
            ax.clear()
            index = _history_index(len(result.value_history), frame, frame_count)
            values = result.value_history[index]
            policy = env.greedy_policy(values, gamma)
            path = env.extract_path(policy)
            valid_path = path if env.validate_path(path) else None
            image = _draw_grid(
                ax,
                env,
                values,
                policy,
                valid_path,
                f"{result.name}\nsweep {index}/{len(result.value_history) - 1}",
                norm,
                show_numbers=False,
            )
            artists.append(image)
        figure.suptitle("Value propagation and the current greedy route", fontsize=14)
        return artists

    movie = animation.FuncAnimation(
        figure, update, frames=frame_count, interval=180, blit=False, repeat=True
    )
    movie.save(output_path, writer=_FullFramePillowWriter(fps=5), dpi=110)
    plt.close(figure)


def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return data.astype(float)
    cumulative = np.cumsum(np.insert(data.astype(float), 0, 0.0))
    return (cumulative[window:] - cumulative[:-window]) / window


def save_monte_carlo_summary(
    env: GridWorld, result: MonteCarloResult, output_path: Path
) -> None:
    """Save the learned map plus rolling episode diagnostics."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite_values = result.values[np.isfinite(result.values)]
    norm = colors.Normalize(
        vmin=float(finite_values.min()), vmax=float(max(0.0, finite_values.max()))
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)

    image = _draw_grid(
        axes[0],
        env,
        result.values,
        result.policy,
        result.path,
        (
            "Learned value and greedy policy\n"
            f"{result.episodes:,} episodes · path {result.path_length}"
        ),
        norm,
    )
    colorbar = figure.colorbar(image, ax=axes[0], shrink=0.78, pad=0.02)
    colorbar.set_label("Learned max action value")

    window = min(500, max(1, result.episodes // 20))
    rolling_return = _moving_average(result.episode_returns, window)
    rolling_length = _moving_average(result.episode_lengths, window)
    x_axis = np.arange(window, result.episodes + 1)

    axes[1].plot(x_axis, rolling_return, color=ALGORITHM_COLORS[0], linewidth=1.8)
    axes[1].set_title(f"Episode return ({window}-episode mean)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Discounted return G₀")
    axes[1].grid(True, alpha=0.25)

    rolling_success = _moving_average(result.successes, window)
    axes[2].plot(
        x_axis,
        rolling_success,
        color=ALGORITHM_COLORS[2],
        linewidth=1.8,
        label="success rate",
    )
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Success rate")
    axes[2].grid(True, alpha=0.25)
    axes[2].set_title(f"Goal completion ({window}-episode mean)")

    length_axis = axes[2].twinx()
    length_axis.plot(
        x_axis,
        rolling_length,
        color=ALGORITHM_COLORS[1],
        linewidth=1.2,
        alpha=0.75,
        label="episode length",
    )
    length_axis.set_ylabel("Episode length")

    lines = axes[2].get_lines() + length_axis.get_lines()
    axes[2].legend(lines, [line.get_label() for line in lines], loc="center right")
    figure.suptitle(
        "First-Visit Monte Carlo Control: sampled learning without Bellman backups",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
