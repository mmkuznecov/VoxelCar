"""Closed-loop evaluation with RL replacing A*.

Pipeline:
    forward camera image
        -> trained OccNet occupancy prediction
        -> trained PPO policy
        -> turn/speed command
        -> simulator step

This is Option A: RL replaces A*.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO

from voxel_car import (
    simulate_episode,
    load_model_from_ckpt,
    episode_summary,
    save_episode_video,
    save_summary_figure,
    generate_scenario,
    SCENARIO_PRESETS,
    compute_fov_mask,
)
from voxel_car.rl import RLPolicyAdapter

DEFAULT_CKPT = "runs/20260419_223836/ckpt_best.pt"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--ckpt",
        default=DEFAULT_CKPT,
        help=f"Path to trained OccNet checkpoint. Default: {DEFAULT_CKPT}",
    )
    p.add_argument(
        "--rl-policy",
        required=True,
        help="Path to trained PPO .zip policy from train_rl.py.",
    )
    p.add_argument(
        "--out-dir",
        default="./closed_loop_rl_runs",
        help="Parent directory for per-run outputs.",
    )

    p.add_argument("--n", type=int, default=4)
    p.add_argument(
        "--preset",
        default="winding",
        choices=list(SCENARIO_PRESETS.keys()),
    )
    p.add_argument("--base-seed", type=int, default=777)

    # Image size fallback. If checkpoint carries image shape, it wins.
    p.add_argument("--img-w", type=int, default=180)
    p.add_argument("--img-h", type=int, default=136)

    # Simulation knobs
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--max-turn-deg", type=float, default=15.0)
    p.add_argument("--min-speed-fraction", type=float, default=0.10)
    p.add_argument("--margin-deg", type=float, default=3.0)

    # Kept for simulator compatibility, but A* does not use these when policy is set.
    p.add_argument("--lookahead-cells", type=int, default=3)
    p.add_argument("--inflate", type=int, default=2)
    p.add_argument("--close-range-cells", type=int, default=3)
    p.add_argument("--no-blind-spot-pessimism", action="store_true")
    p.add_argument("--soft-cost-weight", type=float, default=4.0)
    p.add_argument("--heading-penalty", type=float, default=0.3)
    p.add_argument("--forward-bias", type=float, default=0.7)
    p.add_argument("--no-world-bounds", action="store_true")
    p.add_argument("--world-margin", type=float, default=2.0)
    p.add_argument("--goal-slow-radius", type=float, default=10.0)
    p.add_argument("--goal-slow-min-fraction", type=float, default=0.25)

    # Output
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--no-video", action="store_true")

    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)

    ckpt_path = Path(a.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    rl_path = Path(a.rl_policy)
    if not rl_path.exists() and not Path(str(rl_path) + ".zip").exists():
        raise SystemExit(f"RL policy not found: {rl_path}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(a.out_dir) / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading OccNet checkpoint: {ckpt_path}")
    model, ego_cfg, cam_cfg, image_hw_ckpt, device = load_model_from_ckpt(
        ckpt_path,
        image_h_fallback=a.img_h,
        image_w_fallback=a.img_w,
    )

    H, W = image_hw_ckpt
    print(f"  device={device}")
    print(f"  ego grid: {ego_cfg.shape}")
    print(f"  camera:   {cam_cfg.name}   FOV={cam_cfg.fov}°   yaw={cam_cfg.yaw}°")
    print(f"  image:    {H} × {W}")

    print(f"Loading PPO RL policy: {rl_path}")
    ppo = PPO.load(str(rl_path), device="cpu")

    rl_policy = RLPolicyAdapter(
        model=ppo,
        step_size=float(a.step_size),
        max_turn_deg=float(a.max_turn_deg),
        min_speed_fraction=float(a.min_speed_fraction),
        deterministic=True,
    )

    fov_mask = compute_fov_mask(
        ego_cfg,
        cam_cfg,
        image_w=W,
        image_h=H,
        margin_deg=float(a.margin_deg),
    )
    print(
        f"  FOV mask: {int(fov_mask.sum())}/{fov_mask.size} voxels "
        f"({100*fov_mask.mean():.1f}% in FOV)"
    )

    all_results = []

    for i in range(int(a.n)):
        seed = int(a.base_seed) + i
        scen_name = f"{a.preset}_{i:02d}_seed{seed}_rl"
        print(f"\n=== RL Scenario {i + 1}/{a.n} [{a.preset}] seed={seed} ===")

        scenario = generate_scenario(seed=seed, preset=a.preset, name=scen_name)
        print(
            f"    entry={scenario.entry_xy}  exit={scenario.exit_xy}  "
            f"grid={scenario.grid_shape[:2]}   "
            f"segs={' → '.join(scenario.seg_types)}"
        )

        t0 = time.time()
        episode = simulate_episode(
            scenario,
            model,
            ego_cfg,
            cam_cfg,
            (H, W),
            device,
            fov_mask=fov_mask,
            policy=rl_policy,
            margin_deg=float(a.margin_deg),
            max_steps=int(a.max_steps),
            step_size=float(a.step_size),
            max_turn_deg=float(a.max_turn_deg),
            lookahead_cells=int(a.lookahead_cells),
            inflate=int(a.inflate),
            close_range_cells=int(a.close_range_cells),
            treat_unknown_close_as_obstacle=(not a.no_blind_spot_pessimism),
            soft_cost_weight=float(a.soft_cost_weight),
            heading_penalty=float(a.heading_penalty),
            forward_bias=float(a.forward_bias),
            use_world_bounds=(not a.no_world_bounds),
            world_margin=float(a.world_margin),
            goal_slow_radius=float(a.goal_slow_radius),
            goal_slow_min_fraction=float(a.goal_slow_min_fraction),
            verbose=True,
        )
        sim_s = time.time() - t0

        summary = episode_summary(episode)
        summary["sim_seconds"] = float(sim_s)
        summary["seed"] = seed
        summary["preset"] = a.preset
        summary["controller"] = "ppo_rl_replaces_astar"
        all_results.append(summary)

        stats_path = out_root / f"{scen_name}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(summary, f, indent=2)

        final_png = out_root / f"{scen_name}_final.png"
        save_summary_figure(episode, scenario, cam_cfg, ego_cfg, final_png)
        print(f"    → final snapshot: {final_png.name}")

        if not a.no_video:
            t_vid = time.time()
            video_path = out_root / f"{scen_name}.mp4"
            save_episode_video(
                episode,
                scenario,
                cam_cfg,
                ego_cfg,
                video_path,
                fps=int(a.fps),
            )
            print(
                f"    → video: {video_path.name} "
                f"(render {time.time() - t_vid:.1f}s, sim {sim_s:.1f}s, "
                f"{len(episode.steps)} steps, outcome {episode.outcome})"
            )
        else:
            print(
                f"    (outcome {episode.outcome} in {len(episode.steps)} "
                f"steps, {sim_s:.1f}s)"
            )

    n_success = sum(1 for r in all_results if r["outcome"] == "success")
    n_collide = sum(1 for r in all_results if r["outcome"] == "collision")
    n_stuck = sum(1 for r in all_results if r["outcome"] == "stuck")
    n_oob = sum(1 for r in all_results if r["outcome"] == "oob")
    n_timeout = sum(1 for r in all_results if r["outcome"] == "timeout")

    summary_json = {
        "controller": "ppo_rl_replaces_astar",
        "ckpt": str(ckpt_path),
        "rl_policy": str(rl_path),
        "preset": a.preset,
        "n": int(a.n),
        "image_shape": [H, W],
        "ego_cfg": ego_cfg.to_dict(),
        "camera": cam_cfg.to_dict(),
        "success_rate": n_success / max(len(all_results), 1),
        "outcome_counts": {
            "success": n_success,
            "collision": n_collide,
            "stuck": n_stuck,
            "oob": n_oob,
            "timeout": n_timeout,
        },
        "episodes": all_results,
        "args": vars(a),
    }

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print("\n========================================================")
    print(
        f"RL Results: {n_success}/{len(all_results)} success "
        f"({n_collide} collisions, {n_stuck} stuck, {n_oob} OOB, {n_timeout} timeout)"
    )
    print(f"All outputs in {out_root.resolve()}")


if __name__ == "__main__":
    main()
