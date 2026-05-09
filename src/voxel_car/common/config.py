"""Constants and dataclasses shared across the package.

Conventions (kept consistent *everywhere* in the codebase)
----------------------------------------------------------
World : X east, Y north, Z up  (right-handed)
Camera: OpenCV — X right, Y down, Z forward
Voxel[i, j, k] occupies the cube [i, i+1) × [j, j+1) × [k, k+1)
Car   : forward = heading = (hx, hy); right = (hy, -hx); up = +Z
Yaw   : + yaw rotates a vector toward the car's right side.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional

NUM_CAMERAS = 4

# (R, G, B) overlay colours for the four camera slots.
CAMERA_COLORS: tuple = (
    (60, 200, 230),  # cyan    — Camera 1
    (230, 90, 200),  # magenta — Camera 2
    (250, 170, 60),  # orange  — Camera 3
    (120, 230, 100),  # lime    — Camera 4
)

# Default camera setup: (name, enabled, fwd_m, rgt_m, height_m, yaw_deg, fov_deg)
DEFAULT_CAMERA_SPECS: tuple = (
    ("front", True, 1.5, 0.0, 2.0, 0, 75),
    ("rear", False, -1.5, 0.0, 2.0, 180, 90),
    ("left", False, 0.5, -0.8, 2.0, -90, 110),
    ("right", False, 0.5, 0.8, 2.0, 90, 110),
)


@dataclass
class CameraConfig:
    idx: int
    name: str
    enabled: bool
    fwd: float  # forward offset, metres
    rgt: float  # right offset, metres
    height: float  # height above ground, metres
    yaw: float  # yaw vs car forward, degrees (+ = turn toward car right)
    fov: float  # horizontal field-of-view, degrees

    @property
    def color(self):
        return CAMERA_COLORS[self.idx]

    def to_dict(self):
        d = asdict(self)
        d["color_rgb"] = list(self.color)
        return d


@dataclass
class WorldConfig:
    """Top-level world / terrain configuration.

    Legacy fields (unchanged from before)
    -------------------------------------
    seed, grid_size, max_obstacle_height, road_width, noise_scale,
    shoulder_extra
        Same meaning as the old generator. Defaults match.

    New biome fields (additive — defaults give a slightly richer world
    but stay close in feel to the old one)
    --------------------------------------------------------------
    terrain_power : exponent applied to the noise heightmap before
        scaling. >1 sharpens peaks and flattens valleys → more flat
        ground. The legacy generator used 1.8; we keep that as default.
    noise_octaves : multi-octave value-noise depth. 4 by default.
    flat_threshold : noise values below this are forced to terrain floor
        (= 0). Matches the legacy generator's hard-coded 0.45 threshold.
    sea_level : 0 disables water entirely (legacy default). Set to a
        small positive int (e.g. 1 or 2) to flood low-lying areas. Roads
        will still carve through water as causeways.
    shoreline_thickness : columns within this many voxels above sea level
        render as SAND beach instead of GRASS. Only applies if
        sea_level > 0.
    stone_line_offset : columns at or above this height render as STONE
        on top (rocky peaks). None disables. Set to e.g. ``max_obstacle_height-2``
        to get visible mountain caps.
    dirt_depth : thickness of DIRT layer under the GRASS surface.

    Tree fields
    -----------
    n_trees : 0 disables trees. Positive values scatter trees on grass
        columns away from the road corridor.
    trunk_height_min/max, leaf_radius, tree_min_separation, tree_road_buffer
        Tree placement parameters.
    """

    seed: int = 42
    grid_size: int = 80
    max_obstacle_height: int = 14
    road_width: int = 3
    noise_scale: float = 18.0
    shoulder_extra: int = 2  # extra radius beyond the road where heights fade

    # New biome controls — defaults preserve the old "look" closely.
    terrain_power: float = 1.8
    noise_octaves: int = 4
    flat_threshold: float = 0.45
    sea_level: int = 0
    shoreline_thickness: int = 1
    stone_line_offset: Optional[int] = None
    dirt_depth: int = 2

    # Trees.
    n_trees: int = 0
    trunk_height_min: int = 3
    trunk_height_max: int = 5
    leaf_radius: int = 2
    tree_min_separation: int = 3
    tree_road_buffer: int = 2
    tree_seed: Optional[int] = None  # None → derived from world seed

    @property
    def grid_x(self) -> int:
        return int(self.grid_size)

    @property
    def grid_y(self) -> int:
        return int(self.grid_size)

    @property
    def grid_z(self) -> int:
        # Allow extra headroom for trees if enabled.
        base = int(self.max_obstacle_height) + 3
        if int(self.n_trees) > 0:
            tree_top = int(self.trunk_height_max) + int(self.leaf_radius) + 2
            base = max(base, int(self.max_obstacle_height) + tree_top)
        return base


@dataclass
class TrajectoryConfig:
    seed: int = 1042
    n_segments: int = 6
    noise_amplitude: float = 0.4  # per-waypoint Gaussian noise (voxels)
    smoothing_window: int = 5
    margin: int = 5  # keep waypoints this far from any grid edge


@dataclass
class RenderConfig:
    img_w: int = 180
    img_h: int = 135
    num_frames: int = 40
    fps: int = 10
    bev_display_size: int = 420
    n_samples: int = 200  # ray-march samples per ray


def default_cameras():
    """Instantiate the default CameraConfig list."""
    out = []
    for i, (name, en, fwd, rgt, h, yaw, fov) in enumerate(DEFAULT_CAMERA_SPECS):
        out.append(
            CameraConfig(
                idx=i,
                name=name,
                enabled=en,
                fwd=fwd,
                rgt=rgt,
                height=h,
                yaw=yaw,
                fov=fov,
            )
        )
    return out
