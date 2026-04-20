"""Closed-loop simulator: model + planner drives through a scenario.

At each step:
  (1) Render the forward camera from the *current* car pose.
  (2) Pass the image to the model → ego-frame occupancy logits.
  (3) Pass logits to the planner (which also has the world-frame goal).
  (4) Execute the commanded motion — integrate position with the returned
      heading and step size.
  (5) Check termination (success / collision / stuck / timeout / OOB).

The simulator itself holds ground-truth voxels (needed only for camera
rendering and collision detection). The model and planner only see the
image and their own ego-frame predictions — this is the whole point of the
evaluation.
"""

from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .camera import compute_camera_world_pose, render_camera_view
from .config import CameraConfig
from .planning import plan_next_step, check_collision, PlannerResult
from .train_utils.ego import EgoGridConfig, compute_fov_mask
from .train_utils.model import OccNet

# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    pos: tuple
    heading: tuple
    camera_image: np.ndarray
    pred_occ: np.ndarray
    plan_cells: list
    cost_map: np.ndarray
    ego_goal_cell: tuple
    status: str


@dataclass
class EpisodeRecord:
    scenario_name: str
    outcome: str = "running"
    steps: List[StepRecord] = field(default_factory=list)
    goal_xy: tuple = (0.0, 0.0)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_model_from_ckpt(
    ckpt_path, device=None, image_h_fallback=136, image_w_fallback=180
):
    """Load a trained OccNet + config from a training checkpoint.

    The checkpoint format is the one written by ``train.py`` and includes
    ``ego_cfg``, ``camera``, ``args``, and optionally ``image_shape``.

    Returns ``(model, ego_cfg, cam_cfg, image_hw, device)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # --- ego + camera configs ---
    ego_cfg = EgoGridConfig(**dict(ckpt["ego_cfg"]))
    cam_d = dict(ckpt["camera"])
    cam_d.pop("color_rgb", None)  # not a dataclass field
    cam_cfg = CameraConfig(**cam_d)

    # --- image shape (preferred from ckpt; else fall back) ---
    if "image_shape" in ckpt:
        C, H, W = [int(x) for x in ckpt["image_shape"]]
    else:
        H, W = int(image_h_fallback), int(image_w_fallback)
    image_hw = (H, W)

    # --- model ---
    model = OccNet(d_x=ego_cfg.d_x, d_y=ego_cfg.d_y, d_z=ego_cfg.d_z).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, ego_cfg, cam_cfg, image_hw, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_occupancy(model, image_hwc_u8, device):
    """Run the model on a single ``(H, W, 3)`` uint8 image.

    Returns a ``(Dx, Dy, Dz)`` bool numpy array (sigmoid > 0.5 ≡ logits > 0).
    """
    img = image_hwc_u8.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # (3, H, W)
    t = torch.from_numpy(img).unsqueeze(0).to(device)
    logits = model(t)[0].float().cpu().numpy()
    return logits > 0.0


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------


def simulate_episode(
    scenario,
    model,
    ego_cfg,
    cam_cfg,
    image_hw,
    device,
    fov_mask=None,
    margin_deg=3.0,
    max_steps=100,
    step_size=1.0,
    max_turn_deg=15.0,
    lookahead_cells=3,
    goal_tolerance=3.0,
    stuck_window=15,
    stuck_min_progress_m=4.0,
    n_ray_samples=200,
    # Planner knobs (see src.planning.plan_next_step for details)
    inflate=2,
    close_range_cells=3,
    treat_unknown_close_as_obstacle=True,
    soft_cost_weight=4.0,
    heading_penalty=0.3,
    forward_bias=0.7,
    use_world_bounds=True,
    world_margin=2.0,
    goal_slow_radius=10.0,
    goal_slow_min_fraction=0.25,
    verbose=True,
):
    """Run one closed-loop episode.

    Parameters
    ----------
    scenario     : Scenario — holds voxels + entry/exit + reference traj.
    model        : trained OccNet on ``device``.
    ego_cfg      : EgoGridConfig matching the trained model.
    cam_cfg      : CameraConfig (the forward camera, same as training).
    image_hw     : (H, W) to render the forward camera at.
    fov_mask     : precomputed ``(Dx, Dy, Dz)`` bool mask. If None, computed here.
    margin_deg   : FOV margin for the mask (only used if fov_mask is None).
    stuck_window : consider the car stuck if it makes less than
                   ``stuck_min_progress_m`` in this many consecutive steps.
    """
    H, W = image_hw
    VX, VY, VZ = scenario.voxels.shape
    t_far = max(25.0, 0.85 * max(VX, VY))

    if fov_mask is None:
        fov_mask = compute_fov_mask(
            ego_cfg, cam_cfg, image_w=W, image_h=H, margin_deg=float(margin_deg)
        )

    pos = [float(scenario.entry_xy[0]), float(scenario.entry_xy[1])]
    heading = [float(scenario.entry_heading[0]), float(scenario.entry_heading[1])]

    ep = EpisodeRecord(scenario_name=scenario.name, goal_xy=scenario.exit_xy)
    pos_history = [tuple(pos)]
    # Track closest approach — lets us detect "we reached goal area then
    # started moving away", which is effectively success even if the single-
    # step tolerance check missed it.
    min_d_goal = math.hypot(pos[0] - scenario.exit_xy[0], pos[1] - scenario.exit_xy[1])

    for step in range(int(max_steps)):
        # ---- render forward camera ----
        cam_pos_w, _cam_head, cam_R = compute_camera_world_pose(pos, heading, cam_cfg)
        img = render_camera_view(
            scenario.voxels,
            cam_pos_w,
            cam_R,
            W=W,
            H=H,
            fov_h_deg=cam_cfg.fov,
            t_near=0.2,
            t_far=t_far,
            n_samples=int(n_ray_samples),
        )

        # ---- model forward ----
        pred = predict_occupancy(model, img, device)

        # ---- planner ----
        result = plan_next_step(
            pred,
            ego_cfg,
            tuple(pos),
            tuple(heading),
            scenario.exit_xy,
            fov_mask=fov_mask,
            step_size=float(step_size),
            max_turn_deg=float(max_turn_deg),
            lookahead_cells=int(lookahead_cells),
            inflate=int(inflate),
            close_range_cells=int(close_range_cells),
            treat_unknown_close_as_obstacle=bool(treat_unknown_close_as_obstacle),
            soft_cost_weight=float(soft_cost_weight),
            heading_penalty=float(heading_penalty),
            forward_bias=float(forward_bias),
            world_shape=(VX, VY) if use_world_bounds else None,
            world_margin=float(world_margin),
            goal_slow_radius=float(goal_slow_radius),
            goal_slow_min_fraction=float(goal_slow_min_fraction),
        )

        ep.steps.append(
            StepRecord(
                step=step,
                pos=tuple(pos),
                heading=tuple(heading),
                camera_image=img.copy(),
                pred_occ=pred.copy(),
                plan_cells=list(result.plan_cells),
                cost_map=result.cost_map.copy(),
                ego_goal_cell=tuple(result.ego_goal_cell),
                status=str(result.status),
            )
        )

        # ---- execute motion ----
        nh = result.target_heading_xy
        pos[0] += float(result.step_size) * float(nh[0])
        pos[1] += float(result.step_size) * float(nh[1])
        heading[0] = float(nh[0])
        heading[1] = float(nh[1])
        pos_history.append(tuple(pos))

        # ---- termination checks ----
        d_goal = math.hypot(pos[0] - scenario.exit_xy[0], pos[1] - scenario.exit_xy[1])
        if d_goal < min_d_goal:
            min_d_goal = d_goal

        if d_goal < float(goal_tolerance):
            ep.outcome = "success"
            if verbose:
                print(f"    step {step:3d}  SUCCESS (reached goal, d={d_goal:.2f}m)")
            return ep

        # Overshoot detection: if we were once well within ~1.5× tolerance and
        # we're now moving away (d_goal rising), count it as success. Catches
        # the "drove past the goal" case that single-step tolerance misses when
        # step_size > tolerance or the plan doesn't stop exactly on target.
        overshoot_tol = float(goal_tolerance) * 1.5
        if min_d_goal < overshoot_tol and d_goal > min_d_goal + 0.5:
            ep.outcome = "success"
            if verbose:
                print(
                    f"    step {step:3d}  SUCCESS (closest approach "
                    f"{min_d_goal:.2f}m, now at {d_goal:.2f}m)"
                )
            return ep

        if not (0.0 <= pos[0] < VX and 0.0 <= pos[1] < VY):
            ep.outcome = "oob"
            if verbose:
                print(f"    step {step:3d}  OOB at ({pos[0]:.1f}, {pos[1]:.1f})")
            return ep

        if check_collision(scenario.voxels, pos, heading):
            ep.outcome = "collision"
            if verbose:
                print(f"    step {step:3d}  COLLISION at ({pos[0]:.1f}, {pos[1]:.1f})")
            return ep

        if len(pos_history) > int(stuck_window):
            recent = pos_history[-int(stuck_window) :]
            dist = sum(
                math.hypot(
                    recent[i][0] - recent[i - 1][0], recent[i][1] - recent[i - 1][1]
                )
                for i in range(1, len(recent))
            )
            if dist < float(stuck_min_progress_m):
                ep.outcome = "stuck"
                if verbose:
                    print(
                        f"    step {step:3d}  STUCK (only {dist:.2f}m in last "
                        f"{stuck_window} steps)"
                    )
                return ep

        if verbose and (step % 10 == 0 or step < 3):
            print(
                f"    step {step:3d}  pos=({pos[0]:6.2f}, {pos[1]:6.2f})  "
                f"heading=({heading[0]:+.2f}, {heading[1]:+.2f})  "
                f"plan={result.status}  d_goal={d_goal:5.1f}m"
            )

    ep.outcome = "timeout"
    if verbose:
        print(f"    TIMEOUT after {max_steps} steps (d_goal={d_goal:.1f}m)")
    return ep


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def episode_summary(episode):
    """Compact per-episode summary dict (JSON-friendly, no numpy arrays)."""
    if not episode.steps:
        return {
            "scenario_name": episode.scenario_name,
            "outcome": episode.outcome,
            "n_steps": 0,
        }
    final_pos = episode.steps[-1].pos
    total_dist = 0.0
    prev = episode.steps[0].pos
    for sr in episode.steps[1:]:
        total_dist += math.hypot(sr.pos[0] - prev[0], sr.pos[1] - prev[1])
        prev = sr.pos
    n_no_path = sum(1 for sr in episode.steps if sr.status == "no_path")
    return {
        "scenario_name": episode.scenario_name,
        "outcome": episode.outcome,
        "n_steps": len(episode.steps),
        "final_pos": [float(final_pos[0]), float(final_pos[1])],
        "goal_xy": [float(episode.goal_xy[0]), float(episode.goal_xy[1])],
        "dist_travelled": float(total_dist),
        "n_no_path_steps": int(n_no_path),
    }
