"""Top-down bird's-eye-view rendering.

Includes:
  * terrain colouring from voxel top-heights
  * planned / travelled trajectory polyline
  * rotated-rectangle car glyph + yellow heading indicator
  * semi-transparent frustum wedges with coloured outlines for each camera

World [i = X, j = Y]  →  display [row, col] with row indexed from the top
(so +Y points *up* in the rendered image — north-up map convention).
"""

from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw

from .camera import compute_camera_world_pose

# ---------------------------------------------------------------------------
# Heading / geometry helpers
# ---------------------------------------------------------------------------


def compute_heading(trajectory, idx):
    """Unit heading at waypoint ``idx`` using central differences."""
    n = len(trajectory)
    if idx <= 0:
        d = trajectory[1] - trajectory[0]
    elif idx >= n - 1:
        d = trajectory[-1] - trajectory[-2]
    else:
        d = trajectory[idx + 1] - trajectory[idx - 1]
    return d / (np.linalg.norm(d) + 1e-12)


def _rotate_xy_right_positive(v, angle_rad):
    """Rotate 2D vector v by ``angle_rad``; + angle = toward v's right
    (matches camera yaw convention)."""
    vx, vy = float(v[0]), float(v[1])
    rx, ry = vy, -vx  # right-perp
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return (c * vx + s * rx, c * vy + s * ry)


def _car_rect_corners(car_pos_xy, car_heading_xy, length_m=4.0, width_m=2.4):
    """Four corners of the rotated rectangle representing the car.
    Order: front-right, front-left, rear-left, rear-right."""
    cx, cy = float(car_pos_xy[0]), float(car_pos_xy[1])
    hx, hy = float(car_heading_xy[0]), float(car_heading_xy[1])
    n = math.sqrt(hx * hx + hy * hy) + 1e-12
    hx /= n
    hy /= n
    rx, ry = hy, -hx
    L, W = 0.5 * length_m, 0.5 * width_m
    fr = (cx + L * hx + W * rx, cy + L * hy + W * ry)
    fl = (cx + L * hx - W * rx, cy + L * hy - W * ry)
    rl = (cx - L * hx - W * rx, cy - L * hy - W * ry)
    rr = (cx - L * hx + W * rx, cy - L * hy + W * ry)
    return fr, fl, rl, rr


def _frustum_polygon_world(cam_xy, cam_heading_xy, fov_deg, range_m, n_arc=18):
    """Ground-projected FOV wedge (apex + arc points), in world coords."""
    cx, cy = float(cam_xy[0]), float(cam_xy[1])
    chx, chy = float(cam_heading_xy[0]), float(cam_heading_xy[1])
    n = math.sqrt(chx * chx + chy * chy) + 1e-12
    chx /= n
    chy /= n
    phi = math.radians(fov_deg) * 0.5
    pts = [(cx, cy)]
    for k in range(n_arc + 1):
        a = -phi + (2.0 * phi) * (k / n_arc)
        dx, dy = _rotate_xy_right_positive((chx, chy), a)
        pts.append((cx + range_m * dx, cy + range_m * dy))
    return pts


# ---------------------------------------------------------------------------
# Overlay assembly
# ---------------------------------------------------------------------------


def build_camera_overlays(cams, car_pos_xy, car_heading_xy, include_idxs=None):
    """Translate a list of CameraConfig into overlay dicts consumed by
    ``render_bev``.

    * ``include_idxs=None``  → include only cameras whose ``enabled`` is True.
    * ``include_idxs=set(...)`` → include exactly those indices, ignoring the
      ``enabled`` flag (used by the UI to force-show the selected camera).
    """
    out = []
    for cam in cams:
        if include_idxs is None:
            if not cam.enabled:
                continue
        else:
            if cam.idx not in include_idxs:
                continue
        pos, head, _ = compute_camera_world_pose(car_pos_xy, car_heading_xy, cam)
        out.append(
            {
                "xy": (float(pos[0]), float(pos[1])),
                "heading_xy": (float(head[0]), float(head[1])),
                "fov_deg": cam.fov,
                "color": cam.color,
                "name": cam.name,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render_bev(
    voxels,
    trajectory,
    current_idx,
    car_heading,
    camera_overlays=None,
    display_size=420,
    frustum_range_m=18.0,
):
    """Render a BEV image with terrain, trajectory, frustums, and car.

    ``camera_overlays`` is a list produced by :func:`build_camera_overlays`.
    """
    VX, VY, VZ = voxels.shape

    # Terrain: top-of-column height → colour.
    rev = voxels[:, :, ::-1]
    first_from_top = np.argmax(rev, axis=-1)
    top_z = VZ - 1 - first_from_top
    h_norm = top_z.astype(np.float32) / max(VZ - 1, 1)

    ground_col = np.array([0.50, 0.40, 0.30], dtype=np.float32)
    obs_lo = np.array([0.15, 0.45, 0.15], dtype=np.float32)
    obs_hi = np.array([0.55, 0.85, 0.35], dtype=np.float32)
    is_obst = top_z > 0
    bev = np.where(
        is_obst[..., None],
        obs_lo[None, None, :] + (obs_hi - obs_lo)[None, None, :] * h_norm[..., None],
        ground_col[None, None, :],
    )

    scale = max(1, display_size // max(VX, VY))
    bev_r = np.repeat(np.repeat(bev, scale, axis=0), scale, axis=1)

    disp = np.flipud(np.transpose(bev_r, (1, 0, 2)))
    disp = np.clip(disp * 255.0, 0, 255).astype(np.uint8)
    disp_h, disp_w = disp.shape[:2]

    def to_img(wx, wy):
        col = int(round(float(wx) * scale))
        row = int(round(disp_h - 1 - float(wy) * scale))
        return (col, row)

    base = Image.fromarray(disp).convert("RGBA")

    # ---- Frustum overlay (drawn first, so trajectory/car sit on top) ----
    if camera_overlays:
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for cam in camera_overlays:
            pts_world = _frustum_polygon_world(
                cam["xy"],
                cam["heading_xy"],
                cam["fov_deg"],
                frustum_range_m,
                n_arc=18,
            )
            pts_img = [to_img(px, py) for (px, py) in pts_world]
            r, g, b = cam["color"]
            od.polygon(pts_img, fill=(r, g, b, 55), outline=(r, g, b, 255))
            ax, ay = pts_img[0]
            od.ellipse(
                [ax - 3, ay - 3, ax + 3, ay + 3],
                fill=(r, g, b, 255),
                outline=(255, 255, 255, 255),
            )
        base = Image.alpha_composite(base, overlay)

    draw = ImageDraw.Draw(base)

    # ---- Trajectory (full planned path + travelled portion) ----
    pts_all = [to_img(w[0], w[1]) for w in trajectory]
    if len(pts_all) >= 2:
        draw.line(pts_all, fill=(255, 215, 0, 255), width=2)
        past = pts_all[: int(current_idx) + 1]
        if len(past) > 1:
            draw.line(past, fill=(235, 70, 70, 255), width=4)

    # ---- Car as rotated rectangle + heading line ----
    cx, cy = float(trajectory[current_idx, 0]), float(trajectory[current_idx, 1])
    hx, hy = float(car_heading[0]), float(car_heading[1])
    fr, fl, rl, rr = _car_rect_corners((cx, cy), (hx, hy), length_m=4.0, width_m=2.4)
    poly = [to_img(*fr), to_img(*fl), to_img(*rl), to_img(*rr)]
    draw.polygon(poly, fill=(235, 35, 35, 255), outline=(255, 255, 255, 255))
    front_mid = (0.5 * (fr[0] + fl[0]), 0.5 * (fr[1] + fl[1]))
    draw.line([to_img(cx, cy), to_img(*front_mid)], fill=(255, 255, 0, 255), width=3)

    return np.array(base.convert("RGB"))
