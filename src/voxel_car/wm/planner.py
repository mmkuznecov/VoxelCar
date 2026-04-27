"""Cross-Entropy-Method planner in the JEPA latent space."""

from __future__ import annotations

import math
import numpy as np
import torch

from ..rendering.camera import compute_camera_world_pose, render_camera_view


@torch.no_grad()
def render_goal_image(scen, cam_cfg, H, W, n_ray_samples=150):
    """Render a 'goal image' from the scenario exit, looking toward the last
    reference-trajectory waypoints. Used to produce z_goal."""
    ref = np.asarray(scen.reference_trajectory, dtype=np.float32)
    gx, gy = float(scen.exit_xy[0]), float(scen.exit_xy[1])

    # Heading: average of last few reference deltas, so the goal image faces
    # the direction the car would naturally arrive.
    k = min(4, len(ref) - 1)
    d = ref[-1] - ref[-k]
    nrm = math.hypot(float(d[0]), float(d[1])) + 1e-12
    head = (float(d[0]) / nrm, float(d[1]) / nrm)

    # Stand slightly before the goal point, looking forward -- matches the
    # geometry the car will be in when it arrives.
    pos = np.array([gx - 0.5 * head[0], gy - 0.5 * head[1]], dtype=np.float32)

    VX, VY, _VZ = scen.voxels.shape
    t_far = max(25.0, 0.85 * max(VX, VY))
    cam_pos_w, _, cam_R = compute_camera_world_pose(pos, head, cam_cfg)
    img = render_camera_view(
        scen.voxels,
        cam_pos_w,
        cam_R,
        W=W,
        H=H,
        fov_h_deg=cam_cfg.fov,
        t_near=0.2,
        t_far=t_far,
        n_samples=int(n_ray_samples),
    )
    return img  # (H, W, 3) uint8


class WMCEMPlanner:
    """Cross-Entropy-Method over action sequences in latent space."""

    def __init__(
        self,
        model,
        horizon: int = 5,
        n_samples: int = 256,
        n_elites: int = 32,
        n_iters: int = 4,
        init_std: float = 0.7,
        std_floor: float = 0.1,
        alpha: float = 0.1,
        device: str = "cpu",
    ):
        self.model = model
        self.device = torch.device(device)
        self.horizon = int(horizon)
        self.n_samples = int(n_samples)
        self.n_elites = int(n_elites)
        self.n_iters = int(n_iters)
        self.init_std = float(init_std)
        self.std_floor = float(std_floor)
        self.alpha = float(alpha)

    @torch.no_grad()
    def plan(self, z0, z_goal, mean_init=None):
        """
        z0, z_goal: (D,) torch tensors.
        Returns: (H, 2) numpy best action sequence.
        """
        H = self.horizon
        mean = (
            torch.zeros((H, 2), device=self.device)
            if mean_init is None
            else torch.as_tensor(mean_init, device=self.device, dtype=torch.float32)
        )
        std = torch.full((H, 2), self.init_std, device=self.device)

        z0 = z0.to(self.device).unsqueeze(0).expand(self.n_samples, -1).contiguous()
        z_goal_b = z_goal.to(self.device).unsqueeze(0)

        best_seq = mean.clone()
        for _ in range(self.n_iters):
            noise = torch.randn(self.n_samples, H, 2, device=self.device) * std
            actions = (mean.unsqueeze(0) + noise).clamp(-1.0, 1.0)
            preds = self.model.rollout(z0, actions)  # (S, H, D)
            final = preds[:, -1, :]  # (S, D)
            cost = (final - z_goal_b).pow(2).sum(dim=-1)  # (S,)

            elites = torch.topk(cost, self.n_elites, largest=False).indices
            elite_acts = actions[elites]  # (E, H, 2)
            new_mean = elite_acts.mean(dim=0)
            new_std = elite_acts.std(dim=0).clamp_min(self.std_floor)

            mean = self.alpha * mean + (1.0 - self.alpha) * new_mean
            std = self.alpha * std + (1.0 - self.alpha) * new_std
            best_seq = new_mean

        return best_seq.clamp(-1.0, 1.0).cpu().numpy()
