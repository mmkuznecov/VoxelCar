"""Evaluate an RL policy in the fast oracle-BEV environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from src.rl_env import VoxelCarRLEnv


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to PPO .zip model.")
    p.add_argument(
        "--preset",
        default="easy",
        choices=["easy", "winding", "tall_obstacles", "narrow"],
    )
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--base-seed", type=int, default=9000)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--max-turn-deg", type=float, default=15.0)
    p.add_argument("--min-speed-fraction", type=float, default=0.10)
    p.add_argument("--goal-tolerance", type=float, default=3.0)
    p.add_argument("--out", default="./rl_eval.json")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    model = PPO.load(a.model)

    results = []

    for i in range(int(a.n)):
        seed = int(a.base_seed) + i
        env = VoxelCarRLEnv(
            preset=a.preset,
            base_seed=seed,
            max_steps=a.max_steps,
            step_size=a.step_size,
            max_turn_deg=a.max_turn_deg,
            min_speed_fraction=a.min_speed_fraction,
            goal_tolerance=a.goal_tolerance,
        )

        obs, info = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        final_info = info

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            final_info = info
            done = bool(terminated or truncated)

        results.append(
            {
                "seed": seed,
                "scenario_name": final_info.get("scenario_name"),
                "outcome": final_info.get("outcome", "unknown"),
                "reward": float(total_reward),
                "steps": int(final_info.get("step", -1)),
                "dist_to_goal": float(final_info.get("dist_to_goal", -1.0)),
                "final_pos": final_info.get("pos"),
            }
        )

    counts = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    report = {
        "model": str(a.model),
        "preset": str(a.preset),
        "n": int(a.n),
        "success_rate": counts.get("success", 0) / max(int(a.n), 1),
        "outcome_counts": counts,
        "episodes": results,
    }

    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
