"""RL policy utilities for replacing A* in closed-loop planning.

The RL policy consumes:
    - ego-frame occupancy, usually OccNet-predicted occupancy
    - goal vector in ego coordinates
    - previous action

It outputs:
    - target world heading
    - step size

This module deliberately does not import Stable-Baselines3. The adapter only
expects an object with a `.predict(obs, deterministic=True)` method, which is
what SB3 PPO models expose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..planning import (
    PlannerResult,
    world_to_ego_xy,
    ego_xy_to_cell,
)


def occupancy_to_bev(pred_occ, z_start: int = 1) -> np.ndarray:
    """Collapse (Dx, Dy, Dz) occupancy to a BEV obstacle map.

    Ground layer z=0 is ignored by default because the simulator pins it true.
    """
    occ = np.asarray(pred_occ, dtype=bool)
    if occ.ndim != 3:
        raise ValueError(f"expected pred_occ shape (Dx, Dy, Dz), got {occ.shape}")

    if int(z_start) >= occ.shape[-1]:
        return np.zeros(occ.shape[:2], dtype=bool)

    return occ[:, :, int(z_start) :].any(axis=-1)


def rl_observation_dim(ego_cfg) -> int:
    """Flat observation dimension used by the RL environment and adapter."""
    return int(ego_cfg.d_x) * int(ego_cfg.d_y) + 5


def build_rl_observation(
    pred_occ,
    ego_cfg,
    car_pos,
    car_heading,
    world_goal,
    prev_action: Optional[np.ndarray] = None,
    max_goal_dist: float = 100.0,
    z_start: int = 1,
) -> np.ndarray:
    """Build the flat MLP observation.

    Layout:
        0 : Dx*Dy
            BEV occupancy, flattened, encoded as -1 free / +1 occupied.

        next 3:
            goal_x in ego frame, normalized
            goal_y in ego frame, normalized
            distance-to-goal, normalized

        final 2:
            previous turn command
            previous speed command

    Shape:
        (Dx*Dy + 5,)
    """
    bev = occupancy_to_bev(pred_occ, z_start=z_start).astype(np.float32)

    # Encode binary free/occupied as -1/+1.
    occ_flat = bev.reshape(-1) * 2.0 - 1.0

    gex, gey = world_to_ego_xy(world_goal, car_pos, car_heading)

    forward_range = max(1e-6, float(ego_cfg.d_x) * float(ego_cfg.resolution))
    lateral_range = max(1e-6, 0.5 * float(ego_cfg.d_y) * float(ego_cfg.resolution))

    goal_x = np.clip(float(gex) / forward_range, -1.0, 1.0)
    goal_y = np.clip(float(gey) / lateral_range, -1.0, 1.0)

    d_goal = math.hypot(
        float(world_goal[0]) - float(car_pos[0]),
        float(world_goal[1]) - float(car_pos[1]),
    )
    goal_d = np.clip(d_goal / max(float(max_goal_dist), 1e-6), 0.0, 1.0)

    if prev_action is None:
        prev_action = np.zeros(2, dtype=np.float32)
    else:
        prev_action = np.asarray(prev_action, dtype=np.float32).reshape(-1)
        if prev_action.size < 2:
            prev_action = np.zeros(2, dtype=np.float32)
        else:
            prev_action = prev_action[:2]

    obs = np.concatenate(
        [
            occ_flat.astype(np.float32),
            np.array([goal_x, goal_y, goal_d], dtype=np.float32),
            prev_action.astype(np.float32),
        ],
        axis=0,
    )

    return obs.astype(np.float32)


def _rotate_heading(
    car_heading, turn_cmd: float, max_turn_deg: float
) -> tuple[float, float]:
    """Rotate current heading by a bounded turn command.

    turn_cmd:
        -1 = max right turn
         0 = straight
        +1 = max left turn

    This uses standard world XY rotation: positive is counter-clockwise.
    """
    hx, hy = float(car_heading[0]), float(car_heading[1])
    n = math.hypot(hx, hy) + 1e-12
    hx /= n
    hy /= n

    angle = float(turn_cmd) * math.radians(float(max_turn_deg))
    c, s = math.cos(angle), math.sin(angle)

    nx = hx * c - hy * s
    ny = hx * s + hy * c
    nn = math.hypot(nx, ny) + 1e-12
    return float(nx / nn), float(ny / nn)


def _speed_from_action(
    speed_cmd: float, step_size: float, min_speed_fraction: float
) -> float:
    """Map action speed command from [-1, 1] to [min_frac, 1] * step_size."""
    u = (float(np.clip(speed_cmd, -1.0, 1.0)) + 1.0) * 0.5
    frac = float(min_speed_fraction) + u * (1.0 - float(min_speed_fraction))
    return float(step_size) * float(frac)


def action_to_planner_result(
    action,
    pred_occ,
    ego_cfg,
    car_pos,
    car_heading,
    world_goal,
    step_size: float = 1.0,
    max_turn_deg: float = 15.0,
    min_speed_fraction: float = 0.10,
    z_start: int = 1,
    status: str = "rl",
) -> PlannerResult:
    """Convert a continuous RL action to the PlannerResult expected by the sim.

    Action:
        action[0] = turn command in [-1, 1]
        action[1] = speed command in [-1, 1]
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.size < 2:
        raise ValueError(f"expected action with 2 values, got shape {a.shape}")

    turn_cmd = float(np.clip(a[0], -1.0, 1.0))
    speed_cmd = float(np.clip(a[1], -1.0, 1.0))

    new_heading = _rotate_heading(
        car_heading=car_heading,
        turn_cmd=turn_cmd,
        max_turn_deg=max_turn_deg,
    )
    eff_step = _speed_from_action(
        speed_cmd=speed_cmd,
        step_size=step_size,
        min_speed_fraction=min_speed_fraction,
    )

    cost_map = occupancy_to_bev(pred_occ, z_start=z_start)

    # For visualization only: show where the global goal projects into ego grid.
    gex, gey = world_to_ego_xy(world_goal, car_pos, car_heading)
    ego_goal_cell = ego_xy_to_cell(gex, gey, ego_cfg)

    return PlannerResult(
        target_heading_xy=new_heading,
        step_size=eff_step,
        plan_cells=[],
        ego_goal_cell=ego_goal_cell,
        cost_map=cost_map,
        status=status,
    )


@dataclass
class RLPolicyAdapter:
    """Adapter that lets an SB3 policy replace A* inside simulate_episode()."""

    model: object
    step_size: float = 1.0
    max_turn_deg: float = 15.0
    min_speed_fraction: float = 0.10
    max_goal_dist: float = 100.0
    deterministic: bool = True
    z_start: int = 1

    def __post_init__(self):
        self.prev_action = np.zeros(2, dtype=np.float32)

    def reset(self):
        self.prev_action = np.zeros(2, dtype=np.float32)

    def act(self, pred_occ, ego_cfg, car_pos, car_heading, world_goal) -> PlannerResult:
        obs = build_rl_observation(
            pred_occ=pred_occ,
            ego_cfg=ego_cfg,
            car_pos=car_pos,
            car_heading=car_heading,
            world_goal=world_goal,
            prev_action=self.prev_action,
            max_goal_dist=self.max_goal_dist,
            z_start=self.z_start,
        )

        action, _state = self.model.predict(obs, deterministic=bool(self.deterministic))
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        self.prev_action = action.copy()

        return action_to_planner_result(
            action=action,
            pred_occ=pred_occ,
            ego_cfg=ego_cfg,
            car_pos=car_pos,
            car_heading=car_heading,
            world_goal=world_goal,
            step_size=self.step_size,
            max_turn_deg=self.max_turn_deg,
            min_speed_fraction=self.min_speed_fraction,
            z_start=self.z_start,
            status="rl",
        )
