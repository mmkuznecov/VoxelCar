"""Training artifacts for RL runs.

Outputs:
    episode_metrics.jsonl
        One JSON record per completed episode.

    training_summary.json
        Final aggregate counts and rates.

    plots/
        reward_curve.png
        episode_length_curve.png
        outcome_rates.png
        outcome_counts.png
        distance_to_goal.png

This module is intentionally independent from the environment. It only consumes
the `info` dictionaries emitted by Gymnasium/SB3 vectorized rollouts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

_OUTCOMES = ("success", "collision", "oob", "stuck", "timeout", "running", "unknown")


def _as_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _as_int(x, default=-1) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    window = max(1, int(window))
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(arr.size):
        lo = max(0, i + 1 - window)
        chunk = arr[lo : i + 1]
        if np.isfinite(chunk).any():
            out[i] = np.nanmean(chunk)
        else:
            out[i] = np.nan
    return out


def _rolling_rate(outcomes: Sequence[str], target: str, window: int) -> np.ndarray:
    window = max(1, int(window))
    out = np.zeros(len(outcomes), dtype=np.float64)
    for i in range(len(outcomes)):
        lo = max(0, i + 1 - window)
        chunk = outcomes[lo : i + 1]
        out[i] = sum(o == target for o in chunk) / max(len(chunk), 1)
    return out


def _write_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_episode_metrics(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_episode_metrics(rows: Sequence[dict]) -> dict:
    outcomes = [str(r.get("outcome", "unknown")) for r in rows]
    counts = dict(Counter(outcomes))
    n = len(rows)

    rewards = [_as_float(r.get("reward")) for r in rows]
    lengths = [_as_float(r.get("length")) for r in rows]
    final_dist = [_as_float(r.get("dist_to_goal")) for r in rows]

    summary = {
        "n_episodes": int(n),
        "outcome_counts": counts,
        "success_rate": counts.get("success", 0) / max(n, 1),
        "collision_rate": counts.get("collision", 0) / max(n, 1),
        "oob_rate": counts.get("oob", 0) / max(n, 1),
        "timeout_rate": counts.get("timeout", 0) / max(n, 1),
        "reward_mean": float(np.nanmean(rewards)) if n else None,
        "reward_last_100_mean": float(np.nanmean(rewards[-100:])) if n else None,
        "length_mean": float(np.nanmean(lengths)) if n else None,
        "length_last_100_mean": float(np.nanmean(lengths[-100:])) if n else None,
        "dist_to_goal_mean": float(np.nanmean(final_dist)) if n else None,
        "dist_to_goal_last_100_mean": (
            float(np.nanmean(final_dist[-100:])) if n else None
        ),
    }
    return summary


def write_training_summary(rows: Sequence[dict], out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_episode_metrics(rows)
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_training_plots(
    rows: Sequence[dict],
    out_dir: str | Path,
    rolling_window: int = 100,
) -> None:
    """Write PNG plots from episode-level rows.

    Safe to call repeatedly during training.
    """
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    ep = np.arange(1, len(rows) + 1)
    timesteps = np.asarray(
        [_as_int(r.get("timestep"), i + 1) for i, r in enumerate(rows)]
    )
    rewards = np.asarray([_as_float(r.get("reward")) for r in rows], dtype=np.float64)
    lengths = np.asarray([_as_float(r.get("length")) for r in rows], dtype=np.float64)
    dist = np.asarray(
        [_as_float(r.get("dist_to_goal")) for r in rows], dtype=np.float64
    )
    outcomes = [str(r.get("outcome", "unknown")) for r in rows]

    # 1. Reward curve
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ep, rewards, linewidth=0.8, alpha=0.35, label="episode reward")
    ax.plot(
        ep,
        _rolling_mean(rewards, rolling_window),
        linewidth=2.0,
        label=f"rolling mean ({rolling_window})",
    )
    ax.set_title("RL training reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "reward_curve.png", dpi=130)
    plt.close(fig)

    # 2. Episode length curve
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ep, lengths, linewidth=0.8, alpha=0.35, label="episode length")
    ax.plot(
        ep,
        _rolling_mean(lengths, rolling_window),
        linewidth=2.0,
        label=f"rolling mean ({rolling_window})",
    )
    ax.set_title("Episode length")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "episode_length_curve.png", dpi=130)
    plt.close(fig)

    # 3. Outcome rolling rates
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for outcome in ("success", "collision", "oob", "timeout", "stuck"):
        ax.plot(
            ep,
            _rolling_rate(outcomes, outcome, rolling_window),
            linewidth=2.0,
            label=outcome,
        )
    ax.set_title("Rolling outcome rates")
    ax.set_xlabel("episode")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "outcome_rates.png", dpi=130)
    plt.close(fig)

    # 4. Final outcome counts
    counts = Counter(outcomes)
    labels = [o for o in _OUTCOMES if counts.get(o, 0) > 0]
    values = [counts[o] for o in labels]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values)
    ax.set_title("Final outcome counts")
    ax.set_xlabel("outcome")
    ax.set_ylabel("episodes")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "outcome_counts.png", dpi=130)
    plt.close(fig)

    # 5. Distance-to-goal
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ep, dist, linewidth=0.8, alpha=0.35, label="final distance")
    ax.plot(
        ep,
        _rolling_mean(dist, rolling_window),
        linewidth=2.0,
        label=f"rolling mean ({rolling_window})",
    )
    ax.set_title("Final distance to goal")
    ax.set_xlabel("episode")
    ax.set_ylabel("distance")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "distance_to_goal.png", dpi=130)
    plt.close(fig)

    # 6. Success rate by timestep, useful when vectorized episode ordering is uneven.
    success = np.asarray([1.0 if o == "success" else 0.0 for o in outcomes])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(timesteps, _rolling_mean(success, rolling_window), linewidth=2.0)
    ax.set_title("Rolling success rate by environment step")
    ax.set_xlabel("SB3 timestep")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "success_rate_by_timestep.png", dpi=130)
    plt.close(fig)


class RLMetricsCallback(BaseCallback):
    """Collect episode outcomes/rewards and periodically write plots.

    This expects each environment info dict to include:
        outcome
        dist_to_goal

    If the env is wrapped with stable_baselines3.common.monitor.Monitor,
    `info["episode"]` will also contain reward and length.
    """

    def __init__(
        self,
        out_dir: str | Path,
        rolling_window: int = 100,
        plot_freq: int = 25_000,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.out_dir = Path(out_dir)
        self.metrics_path = self.out_dir / "episode_metrics.jsonl"
        self.rolling_window = int(rolling_window)
        self.plot_freq = int(plot_freq)
        self.rows: list[dict] = []
        self._last_plot_step = 0

    def _on_training_start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Start clean for each run.
        if self.metrics_path.exists():
            self.metrics_path.unlink()

        self.rows = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for env_idx, info in enumerate(infos):
            done = bool(dones[env_idx]) if env_idx < len(dones) else False
            if not done:
                continue

            ep_info = info.get("episode", {}) or {}
            outcome = str(info.get("outcome", ep_info.get("outcome", "unknown")))

            row = {
                "episode": len(self.rows) + 1,
                "timestep": int(self.num_timesteps),
                "env_idx": int(env_idx),
                "outcome": outcome,
                "reward": _as_float(ep_info.get("r", info.get("reward"))),
                "length": _as_int(ep_info.get("l", info.get("step"))),
                "time": _as_float(ep_info.get("t")),
                "dist_to_goal": _as_float(
                    info.get("dist_to_goal", ep_info.get("dist_to_goal"))
                ),
                "progress": _as_float(info.get("progress", ep_info.get("progress"))),
                "step_size": _as_float(info.get("step_size", ep_info.get("step_size"))),
                "scenario_name": info.get("scenario_name"),
                "preset": info.get("preset"),
                "final_pos": info.get("pos"),
            }

            self.rows.append(row)
            _write_jsonl(self.metrics_path, row)

        if self.rows:
            recent = self.rows[-self.rolling_window :]
            recent_outcomes = [str(r.get("outcome", "unknown")) for r in recent]
            recent_rewards = [_as_float(r.get("reward")) for r in recent]
            recent_lengths = [_as_float(r.get("length")) for r in recent]
            recent_dist = [_as_float(r.get("dist_to_goal")) for r in recent]

            self.logger.record("custom/episodes", len(self.rows))
            self.logger.record(
                "custom/success_rate",
                sum(o == "success" for o in recent_outcomes)
                / max(len(recent_outcomes), 1),
            )
            self.logger.record(
                "custom/collision_rate",
                sum(o == "collision" for o in recent_outcomes)
                / max(len(recent_outcomes), 1),
            )
            self.logger.record(
                "custom/oob_rate",
                sum(o == "oob" for o in recent_outcomes) / max(len(recent_outcomes), 1),
            )
            self.logger.record(
                "custom/timeout_rate",
                sum(o == "timeout" for o in recent_outcomes)
                / max(len(recent_outcomes), 1),
            )
            self.logger.record("custom/reward_mean", float(np.nanmean(recent_rewards)))
            self.logger.record("custom/length_mean", float(np.nanmean(recent_lengths)))
            self.logger.record(
                "custom/dist_to_goal_mean", float(np.nanmean(recent_dist))
            )

        if (
            self.plot_freq > 0
            and self.num_timesteps - self._last_plot_step >= self.plot_freq
            and self.rows
        ):
            write_training_plots(
                self.rows,
                self.out_dir,
                rolling_window=self.rolling_window,
            )
            write_training_summary(self.rows, self.out_dir)
            self._last_plot_step = int(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        if self.rows:
            write_training_plots(
                self.rows,
                self.out_dir,
                rolling_window=self.rolling_window,
            )
            write_training_summary(self.rows, self.out_dir)
