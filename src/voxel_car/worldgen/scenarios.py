"""Closed-loop evaluation scenarios.

Each scenario is a world with an entry point, an exit point, and a reference
trajectory (used only for visualisation — the planner never sees it).

The scenarios are deliberately *more complex* than the training distribution:
narrower roads, more segments, taller obstacles. This stresses the learned
perception model's generalisation and gives the planner harder navigation
problems. Success rate at increasing complexity is a good proxy for how
confidently the model generalises.

New biome-aware presets
-----------------------
The presets below now expose the new biome / tree controls that
``WorldConfig`` accepts. Two new presets are added:

  * ``coastal``      — terrain rises out of the sea, sandy shoreline.
  * ``forest``       — heavily wooded plain with scattered hills.

The legacy four (easy / winding / tall_obstacles / narrow) keep their old
look (no water, no trees, no stone peaks) so existing checkpoints still
have a familiar evaluation distribution. To use the richer biome features
with an existing preset, just override fields on the returned scenario's
config and rebuild.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..common.config import WorldConfig, TrajectoryConfig
from .trajectory import build_trajectory
from .world import build_world_from_config

# ---------------------------------------------------------------------------
# Presets (difficulty ladder)
# ---------------------------------------------------------------------------

SCENARIO_PRESETS: dict = {
    # --- Legacy presets ----------------------------------------------------
    # These keep the *occupancy* identical to the old generator (same
    # heightmap, same flat_threshold, same road carve) AND ask the renderer
    # NOT to use the biome palette, so models trained on the old brown-and-
    # green visual distribution continue to see the same images at
    # evaluation time. Flip ``biome_render: True`` on any of these if you
    # want to retrain against the richer palette.
    "easy": dict(
        grid_size=80,
        max_obstacle_height=12,
        road_width=3,
        noise_scale=20.0,
        shoulder_extra=2,
        n_segments=4,
        traj_noise=0.3,
        biome_render=False,
    ),
    "winding": dict(
        grid_size=80,
        max_obstacle_height=14,
        road_width=2,
        noise_scale=16.0,
        shoulder_extra=2,
        n_segments=8,
        traj_noise=0.5,
        biome_render=False,
    ),
    "tall_obstacles": dict(
        grid_size=90,
        max_obstacle_height=18,
        road_width=3,
        noise_scale=14.0,
        shoulder_extra=1,
        n_segments=6,
        traj_noise=0.4,
        biome_render=False,
    ),
    "narrow": dict(
        grid_size=80,
        max_obstacle_height=14,
        road_width=2,
        noise_scale=12.0,
        shoulder_extra=1,
        n_segments=7,
        traj_noise=0.45,
        biome_render=False,
    ),
    # --- New biome-aware presets (use the full palette) --------------------
    # coastal — low terrain, water in valleys, beaches lining the road.
    "coastal": dict(
        grid_size=90,
        max_obstacle_height=14,
        road_width=3,
        noise_scale=22.0,
        shoulder_extra=2,
        n_segments=5,
        traj_noise=0.35,
        terrain_power=1.5,
        flat_threshold=0.30,
        sea_level=2,
        shoreline_thickness=2,
        stone_line_offset=None,
        n_trees=0,
        biome_render=True,
    ),
    # forest — lush plain dense with trees the perception model has to see
    # through. Good stress test for OccNet.
    "forest": dict(
        grid_size=80,
        max_obstacle_height=8,
        road_width=3,
        noise_scale=24.0,
        shoulder_extra=2,
        n_segments=6,
        traj_noise=0.4,
        terrain_power=1.5,
        flat_threshold=0.20,
        sea_level=0,
        stone_line_offset=None,
        n_trees=140,
        trunk_height_min=3,
        trunk_height_max=6,
        leaf_radius=2,
        tree_min_separation=3,
        tree_road_buffer=3,
        biome_render=True,
    ),
    # alpine — tall stone peaks above the noise, narrow winding road.
    "alpine": dict(
        grid_size=90,
        max_obstacle_height=20,
        road_width=2,
        noise_scale=14.0,
        shoulder_extra=1,
        n_segments=8,
        traj_noise=0.45,
        terrain_power=1.6,
        flat_threshold=0.30,
        sea_level=0,
        stone_line_offset=12,
        n_trees=40,
        trunk_height_min=2,
        trunk_height_max=4,
        leaf_radius=2,
        tree_min_separation=4,
        tree_road_buffer=2,
        biome_render=True,
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
    voxels: np.ndarray  # (X, Y, Z) bool — backward compatible
    reference_trajectory: np.ndarray
    entry_xy: tuple
    entry_heading: tuple
    exit_xy: tuple
    world_cfg: WorldConfig
    traj_cfg: TrajectoryConfig
    seg_types: list = field(default_factory=list)
    # New: multi-material grid for biome-aware rendering. Optional so
    # callers that don't care about colour stay zero-cost.
    materials: Optional[np.ndarray] = None  # (X, Y, Z) uint8
    tree_positions: list = field(default_factory=list)

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
            f"unknown preset '{preset}'; options: {list(SCENARIO_PRESETS.keys())}"
        )
    p = SCENARIO_PRESETS[preset]
    biome_render = bool(p.get("biome_render", True))

    world_kwargs = dict(
        seed=int(seed),
        grid_size=int(p["grid_size"]),
        max_obstacle_height=int(p["max_obstacle_height"]),
        road_width=int(p["road_width"]),
        noise_scale=float(p["noise_scale"]),
        shoulder_extra=int(p["shoulder_extra"]),
    )

    # Forward any biome / tree fields the preset specifies.
    for biome_field in (
        "terrain_power",
        "noise_octaves",
        "flat_threshold",
        "sea_level",
        "shoreline_thickness",
        "stone_line_offset",
        "dirt_depth",
        "n_trees",
        "trunk_height_min",
        "trunk_height_max",
        "leaf_radius",
        "tree_min_separation",
        "tree_road_buffer",
    ):
        if biome_field in p:
            world_kwargs[biome_field] = p[biome_field]

    world_cfg = WorldConfig(**world_kwargs)
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
    voxels, _heights, materials, trees = build_world_from_config(
        world_cfg, traj, return_materials=True
    )

    # Legacy presets explicitly opt out of biome rendering — keeps the visual
    # distribution identical to the old generator so trained checkpoints
    # don't see novel imagery at evaluation time.
    materials_for_render = materials if biome_render else None

    entry_xy = (float(traj[0, 0]), float(traj[0, 1]))
    exit_xy = (float(traj[-1, 0]), float(traj[-1, 1]))

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
        materials=materials_for_render,
        tree_positions=trees,
    )
