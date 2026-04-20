"""Closed-loop occupancy-planning evaluation.

Generates several scenarios with entry/exit, loads a trained model, and runs
the car through each one using ONLY the model's predictions from the forward
camera (it never sees the ground-truth voxel grid). The planner is pure A*
over the predicted BEV cost map.

Outputs per run:
    closed_loop_runs/<timestamp>/
        summary.json                    aggregated results
        <scenario>.mp4                  per-step dashboard video
        <scenario>_final.png            final-state snapshot
        <scenario>_stats.json           per-episode stats

Usage
-----
    # uses the hardcoded ckpt path below:
    python run_closed_loop.py

    # explicit overrides:
    python run_closed_loop.py --ckpt runs/xxx/ckpt_best.pt \\
        --preset winding --n 4 --base-seed 777

Presets: easy / winding / tall_obstacles / narrow — see src/scenarios.py.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import simulate_episode, load_model_from_ckpt, episode_summary
from src import save_episode_video, save_summary_figure
from src import generate_scenario, SCENARIO_PRESETS
from src import compute_fov_mask

DEFAULT_CKPT = "runs/20260419_223836/ckpt_best.pt"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ckpt",
        default=DEFAULT_CKPT,
        help=f"Path to trained OccNet checkpoint (default: {DEFAULT_CKPT})",
    )
    p.add_argument(
        "--out-dir",
        default="./closed_loop_runs",
        help="Parent directory for per-run outputs.",
    )
    p.add_argument("--n", type=int, default=4, help="Number of scenarios to run.")
    p.add_argument(
        "--preset",
        default="winding",
        choices=list(SCENARIO_PRESETS.keys()),
        help="Scenario preset (difficulty). See src/scenarios.py.",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=777,
        help="Seed for scenario 0; subsequent scenarios use base+i.",
    )

    # Image size — used both for rendering and the FOV mask computation.
    # Defaults match the training generator (180×136).
    p.add_argument("--img-w", type=int, default=180)
    p.add_argument("--img-h", type=int, default=136)

    # Simulation knobs
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument(
        "--step-size", type=float, default=1.0, help="Metres per simulation step."
    )
    p.add_argument(
        "--max-turn-deg",
        type=float,
        default=15.0,
        help="Maximum heading change per step.",
    )
    p.add_argument(
        "--lookahead-cells",
        type=int,
        default=3,
        help="How many cells along the plan to aim at.",
    )
    p.add_argument(
        "--margin-deg",
        type=float,
        default=3.0,
        help="FOV mask angular margin (should match training).",
    )

    # Planner safety / behaviour knobs
    p.add_argument(
        "--inflate",
        type=int,
        default=2,
        help="Obstacle dilation in cells (car half-width safety buffer). "
        "2 ≈ 2m buffer for a 2.4m-wide car. Lower if the car gets stuck.",
    )
    p.add_argument(
        "--close-range-cells",
        type=int,
        default=3,
        help="Rows within this many cells of the car are handled "
        "pessimistically to guard against the camera blind spot.",
    )
    p.add_argument(
        "--no-blind-spot-pessimism",
        action="store_true",
        help="Disable 'treat close-range unknown as obstacle' — use this "
        "if the car gets stuck on wide-open roads.",
    )
    p.add_argument(
        "--soft-cost-weight",
        type=float,
        default=4.0,
        help="Strength of the distance-transform cost field. Higher → "
        "A* hugs the middle of free corridors more strongly.",
    )
    p.add_argument(
        "--heading-penalty",
        type=float,
        default=0.3,
        help="A* extra cost per heading change. Higher = smoother plans.",
    )
    p.add_argument(
        "--forward-bias",
        type=float,
        default=0.7,
        help="Sub-goal selection: 1.0 always straight-ahead, 0.0 aim at "
        "global goal. 0.7 balances progress vs. not yanking the wheel "
        "toward an unseen exit.",
    )

    # World-boundary safety — prevents driving off the map when the camera
    # sees only sky (the model can't detect the edge of the world from its
    # image; we have to tell the planner about world shape directly).
    p.add_argument(
        "--no-world-bounds",
        action="store_true",
        help="Disable world-boundary obstacles in the cost map. "
        "Only useful for debugging; leave ON for safety.",
    )
    p.add_argument(
        "--world-margin",
        type=float,
        default=2.0,
        help="Metres of buffer inside the world edge the car treats "
        "as obstacle. 2m ≈ car half-length.",
    )

    # Goal approach — slow down near the goal to avoid overshoot.
    p.add_argument(
        "--goal-slow-radius",
        type=float,
        default=10.0,
        help="Within this distance of the world goal, step size is "
        "reduced linearly so the car doesn't overshoot.",
    )
    p.add_argument(
        "--goal-slow-min-fraction",
        type=float,
        default=0.25,
        help="Minimum step-size fraction inside the slow radius.",
    )

    # Output
    p.add_argument(
        "--fps", type=int, default=4, help="Video FPS (1 frame = 1 sim step)."
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Skip MP4 generation (only save final-state PNGs).",
    )

    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)

    ckpt_path = Path(a.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(a.out_dir) / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Load the trained model ----
    print(f"Loading checkpoint: {ckpt_path}")
    model, ego_cfg, cam_cfg, image_hw_ckpt, device = load_model_from_ckpt(
        ckpt_path, image_h_fallback=a.img_h, image_w_fallback=a.img_w
    )
    # If the ckpt carries image shape, honour it; else use CLI args.
    H, W = image_hw_ckpt
    if (H, W) != (a.img_h, a.img_w):
        print(f"  ckpt image shape {H}x{W} overrides CLI {a.img_h}x{a.img_w}")
    print(f"  device={device}")
    print(f"  ego grid: {ego_cfg.shape}")
    print(f"  camera:   {cam_cfg.name}   FOV={cam_cfg.fov}°   yaw={cam_cfg.yaw}°")
    print(f"  image:    {H} × {W}")

    # ---- Precompute FOV mask once (constant across scenarios / steps) ----
    fov_mask = compute_fov_mask(
        ego_cfg, cam_cfg, image_w=W, image_h=H, margin_deg=float(a.margin_deg)
    )
    print(
        f"  FOV mask: {int(fov_mask.sum())}/{fov_mask.size} voxels "
        f"({100*fov_mask.mean():.1f}% in FOV)"
    )

    # ---- Run scenarios ----
    all_results = []
    for i in range(int(a.n)):
        seed = int(a.base_seed) + i
        scen_name = f"{a.preset}_{i:02d}_seed{seed}"
        print(f"\n=== Scenario {i+1}/{a.n}  [{a.preset}]  seed={seed} ===")
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
        all_results.append(summary)

        # Per-episode artefacts
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
                episode, scenario, cam_cfg, ego_cfg, video_path, fps=int(a.fps)
            )
            print(
                f"    → video: {video_path.name}   "
                f"(render {time.time() - t_vid:.1f}s, sim {sim_s:.1f}s, "
                f"{len(episode.steps)} steps, outcome {episode.outcome})"
            )
        else:
            print(
                f"    (outcome {episode.outcome}  in {len(episode.steps)} "
                f"steps, {sim_s:.1f}s)"
            )

    # ---- Aggregate summary ----
    n_success = sum(1 for r in all_results if r["outcome"] == "success")
    n_collide = sum(1 for r in all_results if r["outcome"] == "collision")
    n_stuck = sum(1 for r in all_results if r["outcome"] == "stuck")
    n_oob = sum(1 for r in all_results if r["outcome"] == "oob")
    n_timeout = sum(1 for r in all_results if r["outcome"] == "timeout")

    summary_json = {
        "ckpt": str(ckpt_path),
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

    print(f"\n========================================================")
    print(
        f"Results: {n_success}/{len(all_results)} success   "
        f"({n_collide} collisions, {n_stuck} stuck, "
        f"{n_oob} OOB, {n_timeout} timeout)"
    )
    print(f"All outputs in {out_root.resolve()}")


if __name__ == "__main__":
    main()
