from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class LiveViewerConfig:
    enabled: bool = False
    render_every_episodes: int = 20
    fps: int = 12
    trail_len: int = 200
    show_episode_1: bool = True


class LiveGridViewer:
    """
    Lightweight real-time simulator for GridWorld training.
    It renders selected episodes and highlights collision attempts.
    """

    def __init__(self, cfg: LiveViewerConfig):
        self.cfg = cfg
        self._enabled = cfg.enabled
        self._ready = False
        self._render_this_episode = False
        self._path: list[tuple[int, int]] = []
        self._last_collision: tuple[int, int] | None = None
        self._last_collision_type = "none"
        self._last_draw_ts = 0.0

        self._fig = None
        self._ax = None
        self._plt = None

        if self._enabled:
            self._init_backend()

    @property
    def enabled(self) -> bool:
        return self._enabled and self._ready

    def _init_backend(self) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib import patches
        except Exception as exc:
            print(f"[LiveView] disabled: matplotlib unavailable ({exc})")
            self._enabled = False
            return

        self._plt = plt
        self._patches = patches
        self._plt.ion()
        self._fig, self._ax = self._plt.subplots(figsize=(6, 6))
        self._ready = True

    def on_episode_start(self, episode: int, env) -> None:
        if not self.enabled:
            return
        self._render_this_episode = self._should_render_episode(episode)
        self._path = [(env.agent_r, env.agent_c)]
        self._last_collision = None
        self._last_collision_type = "none"
        if self._render_this_episode:
            self._draw(env=env, episode=episode, step=0, reward=0.0, done=False)

    def on_step(
        self,
        episode: int,
        step: int,
        env,
        action: int,
        reward: float,
        done: bool,
        info: dict,
    ) -> None:
        del action
        if not self.enabled or not self._render_this_episode:
            return

        pos = info.get("position", (env.agent_r, env.agent_c))
        self._path.append(tuple(pos))
        if len(self._path) > self.cfg.trail_len:
            self._path = self._path[-self.cfg.trail_len :]

        if info.get("collision", False):
            attempted = info.get("attempted_pos", pos)
            self._last_collision = tuple(attempted)
            self._last_collision_type = str(info.get("collision_type", "unknown"))
        else:
            self._last_collision = None
            self._last_collision_type = "none"

        self._draw(env=env, episode=episode, step=step, reward=reward, done=done)

    def on_episode_end(self, episode: int, env, episode_return: float, success: bool) -> None:
        if not self.enabled or not self._render_this_episode:
            return
        title = (
            f"Episode {episode} done | return={episode_return:+.3f} "
            f"| success={success}"
        )
        self._ax.set_title(title)
        self._plt.pause(0.001)
        self._draw_rate_limit(force=True)
        del env

    def close(self) -> None:
        if not self.enabled:
            return
        self._plt.ioff()
        self._plt.close(self._fig)

    def _should_render_episode(self, episode: int) -> bool:
        if self.cfg.show_episode_1 and episode == 1:
            return True
        interval = max(1, self.cfg.render_every_episodes)
        return episode % interval == 0

    def _draw(self, env, episode: int, step: int, reward: float, done: bool) -> None:
        if not self.enabled:
            return

        self._ax.clear()
        self._ax.set_aspect("equal")
        self._ax.set_xlim(0, env.W)
        self._ax.set_ylim(0, env.H)
        self._ax.invert_yaxis()
        self._ax.set_xticks(range(env.W + 1))
        self._ax.set_yticks(range(env.H + 1))
        self._ax.grid(color="#d0d0d0", linewidth=0.8)

        for r, c in env.obstacles:
            rect = self._patches.Rectangle((c, r), 1, 1, color="#2f2f2f")
            self._ax.add_patch(rect)

        sr, sc = env.start
        gr, gc = env.goal
        self._ax.add_patch(self._patches.Rectangle((sc, sr), 1, 1, color="#4f81bd", alpha=0.7))
        self._ax.add_patch(self._patches.Rectangle((gc, gr), 1, 1, color="#78be20", alpha=0.7))

        if len(self._path) >= 2:
            xs = [c + 0.5 for _, c in self._path]
            ys = [r + 0.5 for r, _ in self._path]
            self._ax.plot(xs, ys, color="#4094ff", linewidth=2.0, alpha=0.85)

        if self._last_collision is not None:
            cr, cc = self._last_collision
            if 0 <= cr < env.H and 0 <= cc < env.W:
                self._ax.add_patch(
                    self._patches.Rectangle(
                        (cc, cr),
                        1,
                        1,
                        fill=False,
                        edgecolor="#c21807",
                        linewidth=2.0,
                    )
                )

        ar, ac = env.agent_r, env.agent_c
        agent = self._patches.Circle((ac + 0.5, ar + 0.5), radius=0.32, color="#f39c12")
        self._ax.add_patch(agent)

        title = (
            f"Episode {episode} | step={step} | reward={reward:+.2f} "
            f"| done={done} | collision={self._last_collision_type}"
        )
        self._ax.set_title(title)
        self._ax.set_xlabel("col")
        self._ax.set_ylabel("row")

        self._plt.tight_layout()
        self._plt.pause(0.001)
        self._draw_rate_limit()

    def _draw_rate_limit(self, force: bool = False) -> None:
        if force:
            return
        fps = max(1, self.cfg.fps)
        min_interval = 1.0 / fps
        now = time.time()
        elapsed = now - self._last_draw_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_draw_ts = time.time()
