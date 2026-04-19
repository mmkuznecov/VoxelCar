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
from dataclasses import dataclass, asdict


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
    seed: int = 42
    grid_size: int = 80
    max_obstacle_height: int = 14
    road_width: int = 3
    noise_scale: float = 18.0
    shoulder_extra: int = 2  # extra radius beyond the road where heights fade

    @property
    def grid_x(self) -> int:
        return int(self.grid_size)

    @property
    def grid_y(self) -> int:
        return int(self.grid_size)

    @property
    def grid_z(self) -> int:
        return int(self.max_obstacle_height) + 3


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
