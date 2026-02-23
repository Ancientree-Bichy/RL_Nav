import argparse
from pathlib import Path

from rlnav.algorithms import build_agent
from rlnav.envs.gridworld import GridWorld, default_grid_config
from rlnav.evaluation import rollout
from rlnav.infra.logger import TrainingLogger
from rlnav.training import train
from rlnav.utils import make_default_run_dir, save_json, set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular RL navigation training entrypoint")
    parser.add_argument("--algo", choices=["dqn", "pg"], default="dqn")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--moving-avg-window", type=int, default=50)
    parser.add_argument("--no-plot", action="store_true")

    parser.add_argument("--max-steps", type=int, default=200)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--target-update", type=int, default=200)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=float, default=0.999)

    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--render-eval", action="store_true")

    return parser


def resolve_algo_defaults(args: argparse.Namespace) -> None:
    if args.episodes is None:
        args.episodes = 1500 if args.algo == "dqn" else 4000
    if args.lr is None:
        args.lr = 1e-3 if args.algo == "dqn" else 3e-4


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    resolve_algo_defaults(args)

    set_global_seed(args.seed)

    env_cfg = default_grid_config(max_steps=args.max_steps)
    env = GridWorld.from_config(env_cfg)

    agent = build_agent(
        args.algo,
        state_dim=env.n_states,
        action_dim=env.n_actions,
        device=args.device,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        warmup=args.warmup,
        target_update=args.target_update,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
    )

    run_dir = args.run_dir or make_default_run_dir(args.algo)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    train_config = {
        "algo": args.algo,
        "episodes": args.episodes,
        "seed": args.seed,
        "device": args.device,
        "max_steps": args.max_steps,
        "hidden_dim": args.hidden_dim,
        "gamma": args.gamma,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "warmup": args.warmup,
        "target_update": args.target_update,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "eps_decay": args.eps_decay,
        "log_every": args.log_every,
        "save_every": args.save_every,
        "moving_avg_window": args.moving_avg_window,
    }

    save_json(run_path / "config.json", {"env": env_cfg.to_dict(), "train": train_config})

    logger = TrainingLogger(run_path, moving_avg_window=args.moving_avg_window)
    train_result = train(
        env=env,
        agent=agent,
        episodes=args.episodes,
        logger=logger,
        algo_name=args.algo,
        env_config=env_cfg.to_dict(),
        train_config=train_config,
        checkpoint_dir=run_path / "checkpoints",
        log_every=args.log_every,
        save_every=args.save_every,
    )

    plot_path = None if args.no_plot else logger.plot()

    print(f"[Done] run_dir={run_path}")
    print(f"[Done] metrics={run_path / 'metrics.csv'}")
    if plot_path is not None:
        print(f"[Done] plot={plot_path}")
    else:
        print("[Done] plot skipped (use --no-plot) or matplotlib unavailable")
    print(f"[Done] checkpoints={train_result['checkpoint_dir']}")

    if not args.skip_eval:
        eval_episodes = max(args.eval_episodes, 1)
        returns = []
        success_count = 0
        for ep in range(1, eval_episodes + 1):
            ep_ret, ep_steps, success = rollout(
                env=env,
                agent=agent,
                max_steps=args.eval_steps,
                render=args.render_eval,
            )
            returns.append(ep_ret)
            success_count += int(success)
            print(
                f"[Eval] ep={ep:3d}/{eval_episodes}, "
                f"return={ep_ret:+.3f}, steps={ep_steps}, success={success}"
            )

        avg_ret = sum(returns) / len(returns)
        suc_rate = success_count / len(returns)
        print(f"[Eval] avg_return={avg_ret:+.3f}, success_rate={suc_rate:.3f}")


if __name__ == "__main__":
    main()
