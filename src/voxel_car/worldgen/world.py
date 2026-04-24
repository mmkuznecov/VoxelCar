"""Voxel-grid terrain generation with trajectory-aware road + shoulder carving."""

from __future__ import annotations
import numpy as np

from ..common.noise import value_noise_2d
from .trajectory import build_trajectory


def build_world(
    grid_x,
    grid_y,
    grid_z,
    seed,
    noise_scale,
    max_obstacle_height,
    road_width,
    trajectory,
    shoulder_extra=2,
):
    """Generate occupancy voxels shaped around the given trajectory.

    The procedure is:

    1. Multi-octave value noise → raw heightmap in [0, 1].
    2. Height shaping: ``h = (hm ** 1.8) * max_height``, with a low-noise
       threshold that flattens plains.
    3. Adaptive carving around the trajectory:
         * strict road:       d ≤ ``road_width``          →  h = 0
         * shoulder fade:     ``road_width`` < d ≤ ``road_width + shoulder_extra``
                              →  h = min(h,  h · (d − road_width) / shoulder_extra)
       This guarantees the trajectory is always drivable *and* the surrounding
       terrain rises smoothly instead of presenting a vertical cliff.
    4. Lift the per-column heightmap into a boolean voxel grid, pinning the
       bottom layer True so the ground is always opaque to rays.

    Returns
    -------
    voxels  : (grid_x, grid_y, grid_z) bool
    heights : (grid_x, grid_y) int32
    """
    gx, gy, gz = int(grid_x), int(grid_y), int(grid_z)

    hm = value_noise_2d(
        (gx, gy), scale=noise_scale, octaves=4, persistence=0.5, seed=int(seed)
    )
    shaped = np.power(hm, 1.8)
    heights = (shaped * int(max_obstacle_height)).astype(np.int32)
    heights[hm < 0.45] = 0

    r = int(road_width)
    R_sh = r + int(max(0, shoulder_extra))

    # Carve road + shoulder. Small vectorised patch per waypoint — N waypoints ×
    # (2·R_sh+1)² cells, which is cheap for our grid sizes.
    for wp in trajectory:
        ix0 = int(round(float(wp[0])))
        iy0 = int(round(float(wp[1])))
        xi0 = max(0, ix0 - R_sh)
        xi1 = min(gx, ix0 + R_sh + 1)
        yi0 = max(0, iy0 - R_sh)
        yi1 = min(gy, iy0 + R_sh + 1)
        if xi0 >= xi1 or yi0 >= yi1:
            continue

        patch = heights[xi0:xi1, yi0:yi1]
        dxg, dyg = np.meshgrid(
            np.arange(xi0, xi1) - ix0,
            np.arange(yi0, yi1) - iy0,
            indexing="ij",
        )
        d2 = (dxg * dxg + dyg * dyg).astype(np.int32)

        inside_road = d2 <= r * r
        in_shoulder = (~inside_road) & (d2 <= R_sh * R_sh)

        patch[inside_road] = 0

        if R_sh > r and in_shoulder.any():
            d = np.sqrt(d2[in_shoulder]).astype(np.float32)
            fade = (d - float(r)) / float(R_sh - r)  # 0..1
            capped = np.floor(fade * patch[in_shoulder].astype(np.float32)).astype(
                np.int32
            )
            patch[in_shoulder] = np.minimum(patch[in_shoulder], capped)

    # Lift to voxels.
    z_idx = np.arange(gz, dtype=np.int32)
    voxels = z_idx[None, None, :] <= heights[:, :, None]
    voxels[:, :, 0] = True  # ground layer always True
    return voxels, heights


# ---------------------------------------------------------------------------
# Cache: shared by the Gradio preview and video callbacks so dragging a
# camera slider doesn't rebuild the voxel grid.
# ---------------------------------------------------------------------------

_CACHE: dict = {}
_CACHE_MAX = 8


def get_world_and_trajectory(world_cfg, traj_cfg):
    """Return (voxels, trajectory, seg_types) cached on all world + traj params."""
    key = (
        int(world_cfg.seed),
        int(world_cfg.grid_size),
        int(world_cfg.max_obstacle_height),
        int(world_cfg.road_width),
        float(world_cfg.noise_scale),
        int(world_cfg.shoulder_extra),
        int(traj_cfg.seed),
        int(traj_cfg.n_segments),
        float(traj_cfg.noise_amplitude),
        int(traj_cfg.smoothing_window),
        int(traj_cfg.margin),
    )
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
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
        _CACHE[key] = (voxels, traj, seg_types)
    return _CACHE[key]
