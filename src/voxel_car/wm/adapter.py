"""Adapter so WMCEMPlanner drops into simulate_episode like the PPO adapter."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from ..planning import PlannerResult, world_to_ego_xy, ego_xy_to_cell
from ..rendering.camera import compute_camera_world_pose, render_camera_view
from .planner import WMCEMPlanner


def _resize_img_tensor(img_hwc_u8, target_hw):
    """uint8 (H,W,3) -> float tensor (1,3,Ht,Wt) in [0,1]."""
    t = torch.from_numpy(img_hwc_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    Ht, Wt = target_hw
    if t.shape[-2:] != (Ht, Wt):
        t = F.interpolate(t, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return t


class WMPolicyAdapter:
    """Exposes `.act(...)` returning a PlannerResult, compatible with
    simulate_episode(policy=...).

    The adapter re-renders the current camera frame at the WM's image size (so
    the sim's native image resolution doesn't have to match), encodes it,
    CEMs a latent plan against a precomputed goal embedding, and executes the
    first action.
    """

    def __init__(
        self,
        model,
        z_goal,
        cam_cfg,
        *,
        step_size: float = 1.0,
        max_turn_deg: float = 15.0,
        min_speed_fraction: float = 0.10,
        horizon: int = 5,
        n_samples: int = 256,
        n_elites: int = 32,
        n_iters: int = 4,
        init_std: float = 0.7,
        alpha: float = 0.1,
        device: str = "cpu",
        warm_start: bool = True,
    ):
        self.model = model
        self.device = torch.device(device)
        self.cam_cfg = cam_cfg
        self.step_size = float(step_size)
        self.max_turn_deg = float(max_turn_deg)
        self.min_speed_fraction = float(min_speed_fraction)
        self.z_goal = z_goal.to(self.device)
        self.Ht, self.Wt = model.image_hw

        self.planner = WMCEMPlanner(
            model,
            horizon=horizon,
            n_samples=n_samples,
            n_elites=n_elites,
            n_iters=n_iters,
            init_std=init_std,
            alpha=alpha,
            device=device,
        )
        self.warm_start = bool(warm_start)
        self._mean_init = None

    def reset(self):
        self._mean_init = None

    def _cmd_to_planner_result(
        self,
        turn_cmd,
        speed_cmd,
        car_heading,
        pred_occ,
        ego_cfg,
        car_pos,
        world_goal,
    ):
        dtheta = float(turn_cmd) * math.radians(self.max_turn_deg)
        hx, hy = float(car_heading[0]), float(car_heading[1])
        n = math.hypot(hx, hy) + 1e-12
        hx /= n
        hy /= n
        c, s = math.cos(dtheta), math.sin(dtheta)
        new_h = (hx * c - hy * s, hx * s + hy * c)
        nn = math.hypot(new_h[0], new_h[1]) + 1e-12
        new_h = (new_h[0] / nn, new_h[1] / nn)

        u = (float(speed_cmd) + 1.0) * 0.5
        frac = self.min_speed_fraction + u * (1.0 - self.min_speed_fraction)
        eff = self.step_size * frac

        # For visualization: coarse cost map from predicted occupancy, goal in
        # ego cell, empty plan (we have no discrete plan).
        cost_map = (
            pred_occ[:, :, 1:].any(axis=-1)
            if pred_occ.ndim == 3
            else np.zeros((ego_cfg.d_x, ego_cfg.d_y), dtype=bool)
        )
        gex, gey = world_to_ego_xy(world_goal, car_pos, car_heading)
        ego_goal_cell = ego_xy_to_cell(gex, gey, ego_cfg)

        return PlannerResult(
            target_heading_xy=new_h,
            step_size=float(eff),
            plan_cells=[],
            ego_goal_cell=ego_goal_cell,
            cost_map=cost_map,
            status="wm_cem",
        )

    @torch.no_grad()
    def act(
        self,
        pred_occ,
        ego_cfg,
        car_pos,
        car_heading,
        world_goal,
        scen=None,
        image_override=None,
    ):
        """
        Signature matches RLPolicyAdapter.act(), plus optional `image_override`
        so the caller can pass the current camera frame (avoids double-rendering).
        """
        if image_override is not None:
            img = image_override
        else:
            # Render a frame at the WM's input resolution.
            assert scen is not None, "need scenario to render camera frame"
            cam_pos_w, _, cam_R = compute_camera_world_pose(
                car_pos,
                car_heading,
                self.cam_cfg,
            )
            VX, VY, _ = scen.voxels.shape
            t_far = max(25.0, 0.85 * max(VX, VY))
            img = render_camera_view(
                scen.voxels,
                cam_pos_w,
                cam_R,
                W=self.Wt,
                H=self.Ht,
                fov_h_deg=self.cam_cfg.fov,
                t_near=0.2,
                t_far=t_far,
                n_samples=150,
            )

        img_t = _resize_img_tensor(img, (self.Ht, self.Wt)).to(self.device)
        z0 = self.model.encode(img_t).squeeze(0)  # (D,)

        seq = self.planner.plan(z0, self.z_goal, mean_init=self._mean_init)
        turn_cmd, speed_cmd = float(seq[0, 0]), float(seq[0, 1])

        if self.warm_start:
            # Shift the best sequence by one, pad with zeros -> good init next step.
            shifted = np.concatenate(
                [seq[1:], np.zeros((1, 2), dtype=np.float32)],
                axis=0,
            )
            self._mean_init = shifted

        return self._cmd_to_planner_result(
            turn_cmd,
            speed_cmd,
            car_heading,
            pred_occ,
            ego_cfg,
            car_pos,
            world_goal,
        )
