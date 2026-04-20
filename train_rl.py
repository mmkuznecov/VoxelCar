"""Train PPO RL policy for Option A: RL replaces A*.

Artifacts written per run:
    args.json
    ppo_voxel_car_final.zip
    best/best_model.zip
    checkpoints/*.zip
    monitor/*.monitor.csv
    episode_metrics.jsonl
    training_summary.json
    plots/*.png
    tb/...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.rl_env import VoxelCarRLEnv
from src.rl_artifacts import (
    RLMetricsCallback,
    write_training_plots,
    write_training_summary,
)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--out-dir", default="./rl_runs")
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--preset",
        default="easy",
        choices=["easy", "winding", "tall_obstacles", "narrow"],
    )
    p.add_argument("--preset-cycle", action="store_true")
    p.add_argument("--base-seed", type=int, default=777)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--subproc", action="store_true")
    p.add_argument("--steps", type=int, default=1_000_000)

    # Env dynamics
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--max-turn-deg", type=float, default=15.0)
    p.add_argument("--min-speed-fraction", type=float, default=0.10)
    p.add_argument("--goal-tolerance", type=float, default=3.0)

    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip-range", type=float, default=0.2)

    # Artifact / logging
    p.add_argument("--artifact-window", type=int, default=100)
    p.add_argument(
        "--plot-freq",
        type=int,
        default=25_000,
        help="Write PNG plots every N SB3 timesteps. Use 0 to only write at end.",
    )
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--save-freq", type=int, default=100_000)

    # Utility
    p.add_argument("--check-env", action="store_true")

    return p.parse_args(argv)


def _make_env_fn(rank: int, args, run_dir: Path, eval_mode: bool = False):
    def _init():
        seed_offset = 100_000 if eval_mode else 0

        env = VoxelCarRLEnv(
            preset=args.preset,
            base_seed=int(args.base_seed) + seed_offset + rank * 10_000,
            max_steps=int(args.max_steps),
            step_size=float(args.step_size),
            max_turn_deg=float(args.max_turn_deg),
            min_speed_fraction=float(args.min_speed_fraction),
            goal_tolerance=float(args.goal_tolerance),
            preset_cycle=bool(args.preset_cycle),
        )

        monitor_dir = run_dir / "monitor"
        monitor_dir.mkdir(parents=True, exist_ok=True)

        name = "eval" if eval_mode else "train"
        monitor_file = monitor_dir / f"{name}_env_{rank:02d}.monitor.csv"

        # info_keywords makes Monitor persist custom fields into monitor CSV.
        return Monitor(
            env,
            filename=str(monitor_file),
            info_keywords=(
                "outcome",
                "dist_to_goal",
                "progress",
                "step_size",
                "preset",
                "scenario_name",
            ),
        )

    return _init


def main(argv=None):
    a = _parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = a.run_name or f"{stamp}_{a.preset}_ppo"
    out_dir = Path(a.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(a), f, indent=2)

    if a.check_env:
        env = VoxelCarRLEnv(
            preset=a.preset,
            base_seed=a.base_seed,
            max_steps=a.max_steps,
            step_size=a.step_size,
            max_turn_deg=a.max_turn_deg,
            min_speed_fraction=a.min_speed_fraction,
            goal_tolerance=a.goal_tolerance,
            preset_cycle=a.preset_cycle,
        )
        check_env(env, warn=True, skip_render_check=True)
        print("Environment check passed.")

    vec_cls = SubprocVecEnv if (a.subproc and a.n_envs > 1) else DummyVecEnv

    train_env = vec_cls(
        [_make_env_fn(i, a, out_dir, eval_mode=False) for i in range(a.n_envs)]
    )
    eval_env = DummyVecEnv([_make_env_fn(0, a, out_dir, eval_mode=True)])

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, int(a.save_freq) // max(1, int(a.n_envs))),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ppo_voxel_car",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best"),
        log_path=str(out_dir / "eval"),
        eval_freq=max(1, int(a.eval_freq) // max(1, int(a.n_envs))),
        deterministic=True,
        render=False,
    )

    metrics_cb = RLMetricsCallback(
        out_dir=out_dir,
        rolling_window=int(a.artifact_window),
        plot_freq=int(a.plot_freq),
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(out_dir / "tb"),
        learning_rate=float(a.lr),
        n_steps=int(a.n_steps),
        batch_size=int(a.batch_size),
        gamma=float(a.gamma),
        gae_lambda=float(a.gae_lambda),
        ent_coef=float(a.ent_coef),
        clip_range=float(a.clip_range),
    )

    model.learn(
        total_timesteps=int(a.steps),
        callback=[checkpoint_cb, eval_cb, metrics_cb],
        progress_bar=True,
    )

    final_path = out_dir / "ppo_voxel_car_final"
    model.save(final_path)

    # Final artifact pass. The callback also does this, but this makes the
    # behavior explicit and robust if training exits normally after a short run.
    write_training_plots(
        metrics_cb.rows,
        out_dir,
        rolling_window=int(a.artifact_window),
    )
    summary = write_training_summary(metrics_cb.rows, out_dir)

    print(f"Saved final policy to {final_path}.zip")
    print(f"Saved episode metrics to {out_dir / 'episode_metrics.jsonl'}")
    print(f"Saved plots to {out_dir / 'plots'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
