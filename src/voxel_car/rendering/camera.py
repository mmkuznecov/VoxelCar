"""Camera extrinsics + first-person ray-march rendering.

Everything here is in the world / OpenCV-camera convention documented in
``config``. ``CameraConfig`` is the expected input type but a plain dict
with the same fields also works (useful for ad-hoc calls).

Biome-aware rendering
---------------------
``render_camera_view`` accepts an optional ``materials`` argument. When
supplied, the first-hit colour is looked up from ``MATERIAL_PALETTE``
(blue for water, sandy for sand, green for grass, etc.). When omitted
the function falls back to the legacy height-based colouring (green
obstacles + brown checker ground), so all existing call sites remain
correct without modification.
"""

from __future__ import annotations
import math
import numpy as np

from ..common.materials import MATERIAL_PALETTE, Material


def make_camera_R(heading_xy):
    """Return R_cam→world for an OpenCV camera (X right, Y down, Z forward)
    whose optical axis is ``heading_xy`` in the world XY plane."""
    hx, hy = float(heading_xy[0]), float(heading_xy[1])
    n = math.sqrt(hx * hx + hy * hy) + 1e-12
    hx /= n
    hy /= n
    forward = np.array([hx, hy, 0.0], dtype=np.float32)
    right = np.array([hy, -hx, 0.0], dtype=np.float32)
    down = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return np.stack([right, down, forward], axis=1)


def _cam_field(cam, name):
    """Accept either a CameraConfig (attribute access) or dict."""
    return getattr(cam, name) if hasattr(cam, name) else cam[name]


def compute_camera_world_pose(car_pos_xy, car_heading_xy, cam):
    """Return (world_pos_3d, world_heading_xy, R_cam_world) for ``cam``.

    Position:  car + fwd · car_forward + rgt · car_right,  z = height.
    Heading:   cos(yaw) · car_forward + sin(yaw) · car_right
               (+ yaw = rotate heading toward car's right side).
    """
    fwd = float(_cam_field(cam, "fwd"))
    rgt = float(_cam_field(cam, "rgt"))
    height = float(_cam_field(cam, "height"))
    yaw_deg = float(_cam_field(cam, "yaw"))

    hx, hy = float(car_heading_xy[0]), float(car_heading_xy[1])
    n = math.sqrt(hx * hx + hy * hy) + 1e-12
    hx /= n
    hy /= n
    rx, ry = hy, -hx  # car right in world

    cx = float(car_pos_xy[0]) + fwd * hx + rgt * rx
    cy = float(car_pos_xy[1]) + fwd * hy + rgt * ry
    pos = np.array([cx, cy, height], dtype=np.float32)

    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    head_xy = np.array([c * hx + s * rx, c * hy + s * ry], dtype=np.float32)

    return pos, head_xy, make_camera_R(head_xy)


def render_camera_view(
    voxels,
    camera_pos,
    camera_R,
    W,
    H,
    fov_h_deg=75.0,
    t_near=0.2,
    t_far=80.0,
    n_samples=200,
    materials=None,
):
    """Ray-march first-hit renderer.

    Per-pixel rays are constructed in the camera frame, rotated to world,
    then densely sampled at ``n_samples`` depths between ``t_near`` and
    ``t_far``. First True voxel along each ray is the hit.

    Parameters
    ----------
    voxels : (VX, VY, VZ) bool — occupancy / ray-blocking grid. Required.
    materials : (VX, VY, VZ) uint8 — optional. If provided, hit colours
        come from ``MATERIAL_PALETTE``. Otherwise the legacy height-based
        green-and-brown colouring is used.

    Returns an (H, W, 3) uint8 image.
    """
    VX, VY, VZ = voxels.shape

    fov_h = math.radians(fov_h_deg)
    fx = 0.5 * W / math.tan(0.5 * fov_h)
    fy = fx
    cxp = 0.5 * (W - 1.0)
    cyp = 0.5 * (H - 1.0)

    us = (np.arange(W, dtype=np.float32) - cxp) / fx
    vs = (np.arange(H, dtype=np.float32) - cyp) / fy
    uu, vv = np.meshgrid(us, vs)  # (H, W)
    dirs_cam = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (H, W, 3)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_world = dirs_cam @ camera_R.T

    ts = np.linspace(t_near, t_far, int(n_samples), dtype=np.float32)
    pts = (
        camera_pos[None, None, None, :]
        + dirs_world[:, :, None, :] * ts[None, None, :, None]
    )  # (H, W, N, 3)

    ix = np.floor(pts[..., 0]).astype(np.int32)
    iy = np.floor(pts[..., 1]).astype(np.int32)
    iz = np.floor(pts[..., 2]).astype(np.int32)
    in_bounds = (ix >= 0) & (ix < VX) & (iy >= 0) & (iy < VY) & (iz >= 0) & (iz < VZ)
    iix = np.clip(ix, 0, VX - 1)
    iiy = np.clip(iy, 0, VY - 1)
    iiz = np.clip(iz, 0, VZ - 1)
    occ = np.where(in_bounds, voxels[iix, iiy, iiz], False)

    has_hit = occ.any(axis=-1)
    first = np.argmax(occ, axis=-1)  # (H, W)

    # Previous-sample indices for a cheap face-normal guess.
    prev = np.clip(first - 1, 0, int(n_samples) - 1)
    row = np.arange(H)[:, None]
    col = np.arange(W)[None, :]
    hit_ix = iix[row, col, first]
    pr_ix = iix[row, col, prev]
    hit_iy = iiy[row, col, first]
    pr_iy = iiy[row, col, prev]
    hit_iz = iiz[row, col, first]
    pr_iz = iiz[row, col, prev]

    dx = np.sign(hit_ix - pr_ix).astype(np.float32)
    dy = np.sign(hit_iy - pr_iy).astype(np.float32)
    dz = np.sign(hit_iz - pr_iz).astype(np.float32)
    nrm = np.stack([-dx, -dy, -dz], axis=-1)  # opposite of entry
    nrm_len = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = np.where(nrm_len > 0, nrm / (nrm_len + 1e-9), nrm)

    light = np.array([0.4, 0.5, 1.0], dtype=np.float32)
    light /= np.linalg.norm(light)
    lambert = np.clip((nrm * light).sum(axis=-1), 0.0, 1.0)
    shading = 0.45 + 0.55 * lambert

    if materials is not None:
        # ---- Biome-aware colouring ----
        # Look up the palette colour for each hit voxel. Add a subtle
        # checker so flat surfaces have texture (matches the legacy
        # ground-checker look, just generalised across all materials).
        hit_mat = np.where(has_hit, materials[hit_ix, hit_iy, hit_iz], 0).astype(
            np.uint8
        )
        base_rgb = MATERIAL_PALETTE[hit_mat].astype(np.float32) / 255.0  # (H, W, 3)
        checker = ((hit_ix ^ hit_iy) & 1).astype(np.float32) * 0.08 - 0.04
        base_rgb = np.clip(base_rgb + checker[..., None], 0.0, 1.0)
        # Water: keep a flat shading factor so ripples don't show via the
        # lambert term (water's "normal" estimate from the voxel grid is
        # noisy and would otherwise produce dark patches).
        is_water = hit_mat == int(Material.WATER)
        water_shade = 0.85
        shaded = np.where(
            is_water[..., None],
            base_rgb * water_shade,
            base_rgb * shading[..., None],
        )
    else:
        # ---- Legacy height-based colouring ----
        hit_h_norm = hit_iz.astype(np.float32) / max(VZ - 1, 1)
        obst_lo = np.array([0.30, 0.55, 0.30], dtype=np.float32)
        obst_hi = np.array([0.85, 0.95, 0.55], dtype=np.float32)
        obst_rgb = (
            obst_lo[None, None, :]
            + (obst_hi - obst_lo)[None, None, :] * hit_h_norm[..., None]
        )
        ground_mask = hit_iz == 0
        checker = ((hit_ix ^ hit_iy) & 1).astype(np.float32)
        g_base = np.array([0.55, 0.44, 0.32], dtype=np.float32)
        g_alt = np.array([0.45, 0.36, 0.28], dtype=np.float32)
        ground_rgb = (
            g_base[None, None, :] + (g_alt - g_base)[None, None, :] * checker[..., None]
        )
        base_rgb = np.where(ground_mask[..., None], ground_rgb, obst_rgb)
        shaded = base_rgb * shading[..., None]

    # Distance fog toward horizon sky.
    t_hit = ts[first]
    fog = np.clip((t_hit - t_near) / max(t_far - t_near, 1e-6), 0.0, 1.0) ** 1.3
    sky = np.array([0.55, 0.75, 0.95], dtype=np.float32)
    shaded = shaded * (1.0 - fog[..., None]) + sky[None, None, :] * fog[..., None]

    # Pixels that never hit: vertical sky gradient.
    v = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    sky_img = sky[None, None, :] * (0.85 + 0.15 * (1.0 - v[..., None]))
    sky_img = np.broadcast_to(sky_img, (H, W, 3)).copy()

    img = np.where(has_hit[..., None], shaded, sky_img)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)
