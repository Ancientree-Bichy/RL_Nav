from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


class TrainingLogger:
    def __init__(self, run_dir: str | Path, moving_avg_window: int = 50):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.plot_path = self.run_dir / "training_curve.png"
        self.window = moving_avg_window

        self.rows: list[dict[str, Any]] = []
        self.extra_keys: set[str] = set()

    def log_episode(
        self,
        episode: int,
        episode_return: float,
        steps: int,
        success: bool,
        extra_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        extra_metrics = extra_metrics or {}
        self.extra_keys.update(extra_metrics.keys())

        returns = [row["episode_return"] for row in self.rows] + [episode_return]
        successes = [row["success"] for row in self.rows] + [int(success)]

        avg_return = float(np.mean(returns[-self.window :]))
        avg_success = float(np.mean(successes[-self.window :]))

        row: dict[str, Any] = {
            "episode": episode,
            "episode_return": float(episode_return),
            "steps": int(steps),
            "success": int(success),
            f"avg_return_last_{self.window}": avg_return,
            f"avg_success_last_{self.window}": avg_success,
        }
        row.update({k: float(v) for k, v in extra_metrics.items()})
        self.rows.append(row)
        self._flush_csv()
        return row

    def _flush_csv(self) -> None:
        base_keys = [
            "episode",
            "episode_return",
            "steps",
            "success",
            f"avg_return_last_{self.window}",
            f"avg_success_last_{self.window}",
        ]
        extra_keys = sorted(self.extra_keys)
        fieldnames = base_keys + extra_keys

        with self.metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

    def plot(self) -> str | None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        if not self.rows:
            return None

        episodes = np.array([r["episode"] for r in self.rows], dtype=np.int32)
        returns = np.array([r["episode_return"] for r in self.rows], dtype=np.float32)
        success = np.array([r["success"] for r in self.rows], dtype=np.float32)

        def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
            if len(x) == 0:
                return x
            cumsum = np.cumsum(np.insert(x, 0, 0.0))
            ma = (cumsum[w:] - cumsum[:-w]) / float(w)
            pad = np.full(w - 1, ma[0] if len(ma) > 0 else x[0])
            return np.concatenate([pad, ma]) if len(ma) > 0 else x

        ret_ma = moving_avg(returns, min(self.window, len(returns)))
        suc_ma = moving_avg(success, min(self.window, len(success)))

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(episodes, returns, alpha=0.35, label="episode_return")
        axes[0].plot(episodes, ret_ma, linewidth=2, label=f"moving_avg({self.window})")
        axes[0].set_ylabel("Return")
        axes[0].set_title("Training Return")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        axes[1].plot(episodes, success, alpha=0.25, label="success")
        axes[1].plot(episodes, suc_ma, linewidth=2, label=f"moving_avg({self.window})")
        axes[1].set_ylabel("Success")
        axes[1].set_xlabel("Episode")
        axes[1].set_title("Training Success")
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=140)
        plt.close(fig)
        return str(self.plot_path)
