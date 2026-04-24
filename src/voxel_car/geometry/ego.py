"""Ego-frame voxel grid: geometry, FOV mask, world→ego GT extraction.

Axes convention (documented in ``config`` and used throughout the codebase):

    Ego frame:
        ego_x   forward    (along car heading)
        ego_y   right      (+y = passenger side; matches right=(hy,-hx) in world)
        ego_z   up         (same as world z)

    Camera frame (OpenCV):
        x_cam   right
        y_cam   down
        z_cam   forward

Key invariant exploited everywhere: both the ego grid and the cameras are
rigidly attached to the car. Therefore the FOV mask, expressed in ego
coordinates, is the *same for every sample* — compute it once and broadcast.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EgoGridConfig:
    """Ego-frame voxel grid specification.

    The grid is anchored with ego_x starting at 0 (directly in front of car
    centre), ego_y centred at 0 (lateral symmetric), ego_z starting at 0
    (ground). Resolution matches the world grid so the world→ego extraction
    is a clean nearest-neighbour lookup.
    """

    d_x: int = 20  # forward voxels
    d_y: int = 16  # lateral voxels (centred)
    d_z: int = 12  # height voxels (starts at z=0)
    resolution: float = 1.0  # metres per voxel

    @property
    def shape(self):
        return (int(self.d_x), int(self.d_y), int(self.d_z))

    @property
    def num_voxels(self):
        return int(self.d_x) * int(self.d_y) * int(self.d_z)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------


def ego_voxel_centers(cfg):
    """Return ``(Dx, Dy, Dz, 3)`` array of ego-coordinate voxel centres."""
    r = float(cfg.resolution)
    i = np.arange(cfg.d_x, dtype=np.float32)
    j = np.arange(cfg.d_y, dtype=np.float32)
    k = np.arange(cfg.d_z, dtype=np.float32)
    ex = (i + 0.5) * r  # [0, Dx·r)
    ey = (j + 0.5 - cfg.d_y / 2.0) * r  # [-Dy/2·r, +Dy/2·r)
    ez = (k + 0.5) * r  # [0, Dz·r)
    xx, yy, zz = np.meshgrid(ex, ey, ez, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1)  # (Dx, Dy, Dz, 3)


# ---------------------------------------------------------------------------
# FOV mask
# ---------------------------------------------------------------------------


def compute_fov_mask(
    ego_cfg, cam_cfg, image_w, image_h, margin_deg=3.0, include_behind=False
):
    """Boolean mask of ego voxels inside the camera FOV (+ angular margin).

    Since ego grid and camera are both rigidly attached to the car, this
    mask is invariant across samples — compute once, reuse forever.

    Parameters
    ----------
    ego_cfg : EgoGridConfig
    cam_cfg : object with fields ``fwd, rgt, height, yaw, fov`` (degrees for
              yaw and fov). A ``CameraConfig`` works directly.
    image_w, image_h : int
        Used to derive the vertical FOV from the horizontal FOV and the
        image aspect ratio (``tan(fov_v/2) = (H/W)·tan(fov_h/2)``).
    margin_deg : float
        Symmetric angular slack added to both horizontal and vertical FOV.
    include_behind : bool
        If True, include voxels with ``z_cam ≤ 0``. Default False — those
        voxels are unobservable by definition.
    """
    centres = ego_voxel_centers(ego_cfg)  # (Dx, Dy, Dz, 3)

    # Camera pose in ego frame.
    yaw = math.radians(float(cam_cfg.yaw))
    cam_pos = np.array([cam_cfg.fwd, cam_cfg.rgt, cam_cfg.height], dtype=np.float32)
    #   forward in ego: (cos yaw, sin yaw, 0)    (yaw=0 → +ego_x = forward ✓)
    #   right   in ego: (-sin yaw, cos yaw, 0)   (yaw=0 → +ego_y = right   ✓)
    #   down    in ego: (0, 0, -1)               (OpenCV y-down)
    cam_fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
    cam_rgt = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float32)

    v = centres - cam_pos  # (Dx, Dy, Dz, 3)
    z_cam = (v * cam_fwd).sum(-1)
    x_cam = (v * cam_rgt).sum(-1)
    y_cam = -v[..., 2]  # down = -ego_z

    fov_h = math.radians(float(cam_cfg.fov))
    fov_v = 2.0 * math.atan((float(image_h) / float(image_w)) * math.tan(fov_h / 2.0))
    tan_h = math.tan(fov_h / 2.0 + math.radians(margin_deg))
    tan_v = math.tan(fov_v / 2.0 + math.radians(margin_deg))

    eps = 1e-3
    in_front = z_cam > (0.0 if include_behind else eps)
    z_safe = np.where(z_cam > eps, z_cam, eps)  # divide-by-zero guard
    in_h = np.abs(x_cam / z_safe) < tan_h
    in_v = np.abs(y_cam / z_safe) < tan_v
    return (in_front & in_h & in_v).astype(bool)


# ---------------------------------------------------------------------------
# Ground-truth extraction
# ---------------------------------------------------------------------------


def _precomputed_centres_cache():
    """Tiny cache: voxel centres only depend on EgoGridConfig, so cache them."""
    _cache = {}

    def _get(cfg):
        key = (cfg.d_x, cfg.d_y, cfg.d_z, cfg.resolution)
        if key not in _cache:
            _cache[key] = ego_voxel_centers(cfg)
        return _cache[key]

    return _get


_get_centres = _precomputed_centres_cache()


def sample_world_voxels_to_ego(world_voxels, car_pos_xy, car_heading_xy, ego_cfg):
    """Extract a ``(Dx, Dy, Dz)`` bool occupancy tensor by sampling world voxels.

    Each ego voxel centre is rotated/translated into world coordinates, floored
    to the containing world voxel index, and that voxel's occupancy is taken.
    Out-of-bounds ego voxels are returned as False.
    """
    centres = _get_centres(ego_cfg)  # (Dx, Dy, Dz, 3)
    hx, hy = float(car_heading_xy[0]), float(car_heading_xy[1])
    n = math.sqrt(hx * hx + hy * hy) + 1e-12
    hx /= n
    hy /= n
    rx, ry = hy, -hx  # car right in world

    ex = centres[..., 0]
    ey = centres[..., 1]
    ez = centres[..., 2]

    wx = float(car_pos_xy[0]) + ex * hx + ey * rx
    wy = float(car_pos_xy[1]) + ex * hy + ey * ry
    wz = ez

    ix = np.floor(wx).astype(np.int32)
    iy = np.floor(wy).astype(np.int32)
    iz = np.floor(wz).astype(np.int32)

    WX, WY, WZ = world_voxels.shape
    in_b = (ix >= 0) & (ix < WX) & (iy >= 0) & (iy < WY) & (iz >= 0) & (iz < WZ)
    iix = np.clip(ix, 0, WX - 1)
    iiy = np.clip(iy, 0, WY - 1)
    iiz = np.clip(iz, 0, WZ - 1)
    occ = world_voxels[iix, iiy, iiz]
    return np.where(in_b, occ, False)


# ---------------------------------------------------------------------------
# BEV projection of ego grid (for visualisation)
# ---------------------------------------------------------------------------


def ego_to_bev(occupancy_3d):
    """Collapse an ego ``(Dx, Dy, Dz)`` tensor to a ``(Dx, Dy)`` BEV mask
    (any-z occupied). Works on numpy arrays and torch tensors."""
    return (
        occupancy_3d.any(axis=-1)
        if hasattr(occupancy_3d, "any")
        else occupancy_3d.sum(dim=-1).clamp(max=1).bool()
    )
