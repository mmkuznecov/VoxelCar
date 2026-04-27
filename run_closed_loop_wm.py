"""Closed-loop evaluation where the WM's CEM planner replaces A*.

For each scenario:
  1. Load the WM checkpoint.
  2. Render a goal image from the scenario exit and encode it.
  3. Wrap the model in WMPolicyAdapter and pass it as `policy=` to
     simulate_episode -- same hook PPO uses.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from voxel_car import (
    CameraConfig,
    SCENARIO_PRESETS,
    compute_fov_mask,
    generate_scenario,
    simulate_episode,
    episode_summary,
    save_episode_video,
    save_summary_figure,
    load_model_from_ckpt,  # OccNet loader, used only to populate pred_occ
)
from voxel_car.wm import VoxelCarJEPA, WMPolicyAdapter, render_goal_image


def _to_tensor(img, device, Ht, Wt):
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    if t.shape[-2:] != (Ht, Wt):
        t = F.interpolate(t, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return t.to(device)


def _wm_health_check(wm, scen, wm_cam_cfg, Ht, Wt, z_goal, device):
    """Cheap one-shot sanity print: do encodings look numerically healthy and
    does the *initial* car-pose embedding differ from the goal embedding?

    If z_goal and z_init are near-identical or one of them has tiny norm, the
    encoder is broken and CEM can't possibly work -- better to catch it now
    than after 4 minutes of rollouts.
    """
    from voxel_car.rendering.camera import compute_camera_world_pose, render_camera_view

    pos = np.array(scen.entry_xy, dtype=np.float32)
    heading = np.array(scen.entry_heading, dtype=np.float32)
    cam_pos_w, _, cam_R = compute_camera_world_pose(pos, heading, wm_cam_cfg)
    VX, VY, _ = scen.voxels.shape
    t_far = max(25.0, 0.85 * max(VX, VY))
    img = render_camera_view(
        scen.voxels,
        cam_pos_w,
        cam_R,
        W=Wt,
        H=Ht,
        fov_h_deg=wm_cam_cfg.fov,
        t_near=0.2,
        t_far=t_far,
        n_samples=150,
    )
    with torch.no_grad():
        z_init = wm.encode(_to_tensor(img, device, Ht, Wt)).squeeze(0)

    n_init = float(z_init.norm())
    n_goal = float(z_goal.norm())
    diff = float((z_init - z_goal).norm())
    cos = float(
        torch.nn.functional.cosine_similarity(
            z_init.unsqueeze(0), z_goal.unsqueeze(0)
        ).item()
    )
    print(
        f"    sanity: ||z_init||={n_init:.2f}  ||z_goal||={n_goal:.2f}  "
        f"||z_init - z_goal||={diff:.2f}  cos(z_init,z_goal)={cos:+.3f}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wm-ckpt", required=True, help="Path to WM ckpt_best.pt.")
    p.add_argument(
        "--occ-ckpt",
        required=True,
        help="Path to OccNet ckpt (used to populate pred_occ for the viz).",
    )
    p.add_argument("--out-dir", default="./closed_loop_wm_runs")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--preset", default="winding", choices=list(SCENARIO_PRESETS.keys()))
    p.add_argument("--base-seed", type=int, default=777)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--no-video", action="store_true")
    # CEM
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--n-elites", type=int, default=32)
    p.add_argument("--n-iters", type=int, default=4)
    p.add_argument("--init-std", type=float, default=0.7)
    a = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- OccNet (only used so the dashboard panel shows something) ---
    occ_model, ego_cfg, cam_cfg, (H_occ, W_occ), _ = load_model_from_ckpt(
        a.occ_ckpt,
        device=device,
    )

    # --- WM ---
    wm_ckpt = torch.load(a.wm_ckpt, map_location=device, weights_only=False)
    Ht, Wt = wm_ckpt["image_hw"]
    ca = wm_ckpt["args"]
    wm = VoxelCarJEPA(
        image_hw=(Ht, Wt),
        emb_dim=int(ca["emb_dim"]),
        pred_depth=int(ca["pred_depth"]),
        pred_heads=int(ca["pred_heads"]),
    ).to(device)
    wm.load_state_dict(wm_ckpt["model"])
    wm.eval()

    # WM uses its own image size; reuse the OccNet camera geometry since the
    # WM dataset was generated with the same intrinsics.
    wm_cam_cfg = CameraConfig(
        idx=0,
        name="front",
        enabled=True,
        fwd=cam_cfg.fwd,
        rgt=cam_cfg.rgt,
        height=cam_cfg.height,
        yaw=cam_cfg.yaw,
        fov=cam_cfg.fov,
    )

    fov_mask = compute_fov_mask(
        ego_cfg,
        cam_cfg,
        image_w=W_occ,
        image_h=H_occ,
        margin_deg=3.0,
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(a.out_dir) / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i in range(int(a.n)):
        seed = int(a.base_seed) + i
        scen_name = f"{a.preset}_{i:02d}_seed{seed}_wm"
        print(f"\n=== WM-CEM Scenario {i + 1}/{a.n}  seed={seed} ===")
        scen = generate_scenario(seed=seed, preset=a.preset, name=scen_name)

        # Encode goal.
        goal_img = render_goal_image(scen, wm_cam_cfg, Ht, Wt)
        with torch.no_grad():
            z_goal = wm.encode(_to_tensor(goal_img, device, Ht, Wt)).squeeze(0)

        _wm_health_check(wm, scen, wm_cam_cfg, Ht, Wt, z_goal, device)

        adapter = WMPolicyAdapter(
            wm,
            z_goal,
            wm_cam_cfg,
            horizon=int(a.horizon),
            n_samples=int(a.n_samples),
            n_elites=int(a.n_elites),
            n_iters=int(a.n_iters),
            init_std=float(a.init_std),
            device=str(device),
        )

        scen_holder = {"scen": scen}

        class _Policy:
            def reset(self):
                adapter.reset()

            def act(self, pred_occ, ego_cfg, car_pos, car_heading, world_goal):
                return adapter.act(
                    pred_occ,
                    ego_cfg,
                    car_pos,
                    car_heading,
                    world_goal,
                    scen=scen_holder["scen"],
                )

        t0 = time.time()
        ep = simulate_episode(
            scen,
            occ_model,
            ego_cfg,
            cam_cfg,
            (H_occ, W_occ),
            device,
            fov_mask=fov_mask,
            policy=_Policy(),
            margin_deg=3.0,
            max_steps=int(a.max_steps),
            step_size=1.0,
            max_turn_deg=15.0,
            verbose=True,
        )
        sim_s = time.time() - t0

        summary = episode_summary(ep)
        summary.update(
            {
                "sim_seconds": sim_s,
                "seed": seed,
                "preset": a.preset,
                "controller": "wm_cem",
            }
        )
        all_results.append(summary)

        with open(out_root / f"{scen_name}_stats.json", "w") as f:
            json.dump(summary, f, indent=2)
        save_summary_figure(
            ep,
            scen,
            cam_cfg,
            ego_cfg,
            out_root / f"{scen_name}_final.png",
        )
        if not a.no_video:
            save_episode_video(
                ep,
                scen,
                cam_cfg,
                ego_cfg,
                out_root / f"{scen_name}.mp4",
                fps=int(a.fps),
            )

    counts = {
        k: sum(1 for r in all_results if r["outcome"] == k)
        for k in ("success", "collision", "stuck", "oob", "timeout")
    }
    summary_json = {
        "controller": "wm_cem",
        "wm_ckpt": str(a.wm_ckpt),
        "preset": a.preset,
        "n": int(a.n),
        "success_rate": counts["success"] / max(len(all_results), 1),
        "outcome_counts": counts,
        "episodes": all_results,
        "args": vars(a),
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print("\n========================================================")
    print(
        f"WM-CEM Results: {counts['success']}/{len(all_results)} success "
        f"(coll {counts['collision']}, stuck {counts['stuck']}, "
        f"oob {counts['oob']}, timeout {counts['timeout']})"
    )
    print(f"All outputs in {out_root.resolve()}")


if __name__ == "__main__":
    main()
