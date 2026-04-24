"""Closed-loop evaluation scenarios.

Each scenario is a world with an entry point, an exit point, and a reference
trajectory (used only for visualisation — the planner never sees it).

The scenarios are deliberately *more complex* than the training distribution:
narrower roads, more segments, taller obstacles. This stresses the learned
perception model's generalisation and gives the planner harder navigation
problems. Success rate at increasing complexity is a good proxy for how
confidently the model generalises.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np

from ..common.config import WorldConfig, TrajectoryConfig
from .trajectory import build_trajectory
from .world import build_world

# ---------------------------------------------------------------------------
# Presets (difficulty ladder)
# ---------------------------------------------------------------------------

SCENARIO_PRESETS: dict = {
    # easy — close to training distribution
    "easy": dict(
        grid_size=80,
        max_obstacle_height=12,
        road_width=3,
        noise_scale=20.0,
        shoulder_extra=2,
        n_segments=4,
        traj_noise=0.3,
    ),
    # winding — more segments, narrower road
    "winding": dict(
        grid_size=80,
        max_obstacle_height=14,
        road_width=2,
        noise_scale=16.0,
        shoulder_extra=2,
        n_segments=8,
        traj_noise=0.5,
    ),
    # tall — bigger obstacles, sharper terrain
    "tall_obstacles": dict(
        grid_size=90,
        max_obstacle_height=18,
        road_width=3,
        noise_scale=14.0,
        shoulder_extra=1,
        n_segments=6,
        traj_noise=0.4,
    ),
    # narrow — tight road, bumpy terrain
    "narrow": dict(
        grid_size=80,
        max_obstacle_height=14,
        road_width=2,
        noise_scale=12.0,
        shoulder_extra=1,
        n_segments=7,
        traj_noise=0.45,
    ),
}


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    preset: str
    seed: int
    voxels: np.ndarray
    reference_trajectory: np.ndarray
    entry_xy: tuple
    entry_heading: tuple
    exit_xy: tuple
    world_cfg: WorldConfig
    traj_cfg: TrajectoryConfig
    seg_types: list = field(default_factory=list)

    @property
    def grid_shape(self):
        return tuple(int(s) for s in self.voxels.shape)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_scenario(seed, preset="winding", name=None):
    """Instantiate one scenario from a preset.

    The reference trajectory's first waypoint is the entry; the last is the
    exit. The initial heading is aligned with the first couple of waypoints.
    """
    if preset not in SCENARIO_PRESETS:
        raise ValueError(
            f"unknown preset '{preset}'; options: " f"{list(SCENARIO_PRESETS.keys())}"
        )
    p = SCENARIO_PRESETS[preset]
    world_cfg = WorldConfig(
        seed=int(seed),
        grid_size=int(p["grid_size"]),
        max_obstacle_height=int(p["max_obstacle_height"]),
        road_width=int(p["road_width"]),
        noise_scale=float(p["noise_scale"]),
        shoulder_extra=int(p["shoulder_extra"]),
    )
    traj_cfg = TrajectoryConfig(
        seed=int(seed) + 1000,
        n_segments=int(p["n_segments"]),
        noise_amplitude=float(p["traj_noise"]),
    )

    traj, seg_types = build_trajectory(
        world_cfg.grid_x,
        world_cfg.grid_y,
        seed=traj_cfg.seed,
        n_segments=traj_cfg.n_segments,
        noise_amplitude=traj_cfg.noise_amplitude,
        smoothing_window=traj_cfg.smoothing_window,
        margin=traj_cfg.margin,
    )
    voxels, _ = build_world(
        world_cfg.grid_x,
        world_cfg.grid_y,
        world_cfg.grid_z,
        seed=world_cfg.seed,
        noise_scale=world_cfg.noise_scale,
        max_obstacle_height=world_cfg.max_obstacle_height,
        road_width=world_cfg.road_width,
        trajectory=traj,
        shoulder_extra=world_cfg.shoulder_extra,
    )

    entry_xy = (float(traj[0, 0]), float(traj[0, 1]))
    exit_xy = (float(traj[-1, 0]), float(traj[-1, 1]))

    # Initial heading: average of first few path deltas so we don't point at
    # a single noisy waypoint.
    head_lookahead = min(3, len(traj) - 1)
    dx = float(traj[head_lookahead, 0] - traj[0, 0])
    dy = float(traj[head_lookahead, 1] - traj[0, 1])
    hn = math.hypot(dx, dy) + 1e-12
    entry_heading = (dx / hn, dy / hn)

    if name is None:
        name = f"{preset}_seed{int(seed)}"

    return Scenario(
        name=str(name),
        preset=str(preset),
        seed=int(seed),
        voxels=voxels,
        reference_trajectory=traj,
        entry_xy=entry_xy,
        entry_heading=entry_heading,
        exit_xy=exit_xy,
        world_cfg=world_cfg,
        traj_cfg=traj_cfg,
        seg_types=list(seg_types),
    )
