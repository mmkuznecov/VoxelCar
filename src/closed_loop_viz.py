"""Per-step dashboard rendering for closed-loop episodes.

Each rendered frame is a 3-panel figure:

    ┌─────────────────────────┬──────────────────┐
    │                         │  forward camera  │
    │       world BEV         │  (what the car   │
    │  with reference path,   │   actually sees) │
    │  actual trajectory,     ├──────────────────┤
    │  camera frustum,        │  predicted BEV   │
    │  goal marker            │  + planned path  │
    │                         │                  │
    └─────────────────────────┴──────────────────┘

``save_episode_video`` composes every step into an MP4. A per-episode
summary figure (final state only) is also available via ``save_summary_figure``.
"""

from __future__ import annotations
import io
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio
from PIL import Image

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_C_REF_TRAJ = (0.20, 0.80, 0.90)  # cyan — reference path (not seen by car)
_C_ACT_TRAJ = (0.20, 0.95, 0.30)  # lime — actual car path
_C_CAR = (0.95, 0.25, 0.25)  # red
_C_HEADING = (1.00, 0.95, 0.00)
_C_GOAL = (1.00, 0.80, 0.00)  # yellow star
_C_FRUSTUM = (0.35, 0.80, 0.95)

_C_OBST = (0.45, 0.45, 0.48)
_C_FREE = (0.90, 0.90, 0.93)
_C_PLAN = (1.00, 0.85, 0.10)

_BG_OUTCOME = {
    "success": (0.20, 0.60, 0.30),
    "collision": (0.85, 0.25, 0.25),
    "stuck": (0.75, 0.55, 0.15),
    "oob": (0.55, 0.35, 0.75),
    "timeout": (0.40, 0.45, 0.55),
    "running": (0.30, 0.30, 0.35),
}


# ---------------------------------------------------------------------------
# World BEV background (raw terrain from voxels, drawn in matplotlib coords)
# ---------------------------------------------------------------------------


def _world_terrain_bev(voxels):
    """Return a ``(VX, VY, 3)`` float32 RGB image of terrain heights."""
    VX, VY, VZ = voxels.shape
    rev = voxels[:, :, ::-1]
    first_from_top = np.argmax(rev, axis=-1)
    top_z = VZ - 1 - first_from_top
    h_norm = top_z.astype(np.float32) / max(VZ - 1, 1)

    ground = np.array([0.50, 0.40, 0.30], dtype=np.float32)
    obs_lo = np.array([0.15, 0.45, 0.15], dtype=np.float32)
    obs_hi = np.array([0.55, 0.85, 0.35], dtype=np.float32)
    is_obst = top_z > 0
    return np.where(
        is_obst[..., None],
        obs_lo[None, None, :] + (obs_hi - obs_lo)[None, None, :] * h_norm[..., None],
        ground[None, None, :],
    )


def _draw_frustum(
    ax,
    cam_xy,
    cam_heading,
    fov_deg,
    range_m=16.0,
    n_arc=14,
    color=_C_FRUSTUM,
    alpha_face=0.18,
    alpha_edge=0.85,
):
    """Draw a FOV wedge in world coords on ``ax``."""
    chx, chy = float(cam_heading[0]), float(cam_heading[1])
    n = math.hypot(chx, chy) + 1e-12
    chx /= n
    chy /= n
    phi = math.radians(float(fov_deg)) * 0.5
    pts = [(float(cam_xy[0]), float(cam_xy[1]))]
    for k in range(n_arc + 1):
        a = -phi + (2.0 * phi) * (k / n_arc)
        c, s = math.cos(a), math.sin(a)
        dx = c * chx - s * chy
        dy = s * chx + c * chy
        pts.append((cam_xy[0] + range_m * dx, cam_xy[1] + range_m * dy))
    poly = patches.Polygon(
        pts,
        closed=True,
        facecolor=(*color, alpha_face),
        edgecolor=(*color, alpha_edge),
        linewidth=1.0,
    )
    ax.add_patch(poly)


def _draw_car_glyph(ax, pos, heading, length=4.0, width=2.4, color=_C_CAR):
    hx, hy = float(heading[0]), float(heading[1])
    n = math.hypot(hx, hy) + 1e-12
    hx /= n
    hy /= n
    rx, ry = hy, -hx
    L, W = length / 2, width / 2
    fr = (pos[0] + L * hx + W * rx, pos[1] + L * hy + W * ry)
    fl = (pos[0] + L * hx - W * rx, pos[1] + L * hy - W * ry)
    rl = (pos[0] - L * hx - W * rx, pos[1] - L * hy - W * ry)
    rr = (pos[0] - L * hx + W * rx, pos[1] - L * hy + W * ry)
    ax.add_patch(
        patches.Polygon(
            [fr, fl, rl, rr],
            closed=True,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
        )
    )
    front_mid = (0.5 * (fr[0] + fl[0]), 0.5 * (fr[1] + fl[1]))
    ax.plot(
        [pos[0], front_mid[0]], [pos[1], front_mid[1]], color=_C_HEADING, linewidth=1.5
    )


# ---------------------------------------------------------------------------
# Predicted ego BEV panel
# ---------------------------------------------------------------------------


def _draw_pred_bev(ax, step_record, ego_cfg):
    """Cost map + planned path on the ego BEV (forward=up, right=right)."""
    cost = step_record.cost_map
    Dx, Dy = cost.shape
    res = float(ego_cfg.resolution)

    rgb = np.where(
        cost[..., None],
        np.array(_C_OBST, dtype=np.float32)[None, None, :],
        np.array(_C_FREE, dtype=np.float32)[None, None, :],
    )
    # Display with forward=up: flip the Dx axis.
    disp = np.flip(rgb.astype(np.float32), axis=0)
    extent = [-Dy / 2.0 * res, Dy / 2.0 * res, 0.0, Dx * res]
    ax.imshow(
        disp, extent=extent, origin="lower", aspect="equal", interpolation="nearest"
    )

    # Plan (convert cells → ego metric).
    if step_record.plan_cells:
        xs, ys = [], []
        for i, j in step_record.plan_cells:
            ex = (i + 0.5) * res
            ey = (j + 0.5 - Dy / 2.0) * res
            ys.append(ex)
            xs.append(ey)  # imshow: x = right, y = forward
        ax.plot(
            xs,
            ys,
            "o-",
            color=_C_PLAN,
            markersize=3,
            linewidth=2.0,
            markeredgecolor="black",
            markeredgewidth=0.4,
        )

    # Goal cell marker.
    gi, gj = step_record.ego_goal_cell
    gex = (gi + 0.5) * res
    gey = (gj + 0.5 - Dy / 2.0) * res
    ax.plot(gey, gex, "*", color=_C_GOAL, markersize=16, markeredgecolor="black")

    # Car glyph at origin (below the grid).
    ax.add_patch(
        patches.Rectangle((-1.2, -2.4), 2.4, 2.0, facecolor=_C_CAR, edgecolor="white")
    )
    ax.plot([0, 0], [-0.4, 1.0], "-", color=_C_HEADING, linewidth=2)

    ax.set_xlim(extent[0] - 1, extent[1] + 1)
    ax.set_ylim(-3, extent[3] + 1)
    ax.set_xlabel("ego_y (right, m)", fontsize=8)
    ax.set_ylabel("ego_x (forward, m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2)
    title = f"Predicted BEV + plan   (plan status: {step_record.status})"
    ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Dashboard frame
# ---------------------------------------------------------------------------


def render_episode_frame(
    scenario,
    step_record,
    past_positions,
    cam_cfg,
    ego_cfg,
    outcome=None,
    figsize=(13.5, 6.4),
):
    """Render one dashboard frame for a single simulation step."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.35, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.22,
        wspace=0.18,
    )
    ax_world = fig.add_subplot(gs[:, 0])
    ax_cam = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[1, 1])

    # ---- world BEV ----
    terrain = _world_terrain_bev(scenario.voxels)
    VX, VY = terrain.shape[:2]
    # Transpose so row index → y, col index → x; origin='lower' puts +y up.
    ax_world.imshow(
        terrain.transpose(1, 0, 2),
        origin="lower",
        extent=(0, VX, 0, VY),
        interpolation="nearest",
    )

    ref = scenario.reference_trajectory
    ax_world.plot(
        ref[:, 0],
        ref[:, 1],
        color=_C_REF_TRAJ,
        linewidth=1.2,
        alpha=0.65,
        label="reference (hidden from car)",
    )

    if len(past_positions) >= 2:
        pp = np.asarray(past_positions)
        ax_world.plot(
            pp[:, 0], pp[:, 1], color=_C_ACT_TRAJ, linewidth=2.2, label="actual path"
        )

    _draw_frustum(
        ax_world,
        (
            step_record.pos[0] + cam_cfg.fwd * step_record.heading[0],
            step_record.pos[1] + cam_cfg.fwd * step_record.heading[1],
        ),
        step_record.heading,
        cam_cfg.fov,
        range_m=16.0,
    )
    _draw_car_glyph(ax_world, step_record.pos, step_record.heading)

    # Entry / exit markers
    ax_world.plot(
        [scenario.entry_xy[0]],
        [scenario.entry_xy[1]],
        "s",
        color="white",
        markersize=7,
        markeredgecolor="black",
    )
    ax_world.plot(
        [scenario.exit_xy[0]],
        [scenario.exit_xy[1]],
        "*",
        color=_C_GOAL,
        markersize=20,
        markeredgecolor="black",
    )

    ax_world.set_xlim(0, VX)
    ax_world.set_ylim(0, VY)
    ax_world.set_aspect("equal")
    ax_world.set_xlabel("world x (east)", fontsize=8)
    ax_world.set_ylabel("world y (north)", fontsize=8)
    ax_world.tick_params(labelsize=7)
    tag = f"{scenario.name}   step {step_record.step}"
    if outcome is not None:
        tag += f"   [{outcome}]"
    ax_world.set_title(tag, fontsize=10)
    ax_world.legend(loc="lower right", fontsize=7, framealpha=0.85)

    # ---- camera image ----
    ax_cam.imshow(step_record.camera_image)
    ax_cam.set_title("Forward camera (model input)", fontsize=9)
    ax_cam.axis("off")

    # ---- predicted BEV + plan ----
    _draw_pred_bev(ax_pred, step_record, ego_cfg)

    return fig


# ---------------------------------------------------------------------------
# Episode video
# ---------------------------------------------------------------------------


def _fig_to_rgb(fig, dpi=90):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def save_episode_video(
    episode, scenario, cam_cfg, ego_cfg, out_path, fps=4, dpi=80, progress_fn=None
):
    """Render every step to a frame and write an MP4.

    Returns the final frame size (w, h) so callers can pre-allocate if needed.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    past = []
    H0 = W0 = None
    for idx, sr in enumerate(episode.steps):
        past.append(sr.pos)
        outcome = episode.outcome if idx == len(episode.steps) - 1 else None
        fig = render_episode_frame(
            scenario, sr, past, cam_cfg, ego_cfg, outcome=outcome
        )
        fr = _fig_to_rgb(fig, dpi=dpi)
        plt.close(fig)

        # Pad/crop all frames to a consistent size for the encoder.
        if H0 is None:
            H0, W0 = fr.shape[:2]
            # Ensure even dimensions (required by most codecs).
            H0 -= H0 % 2
            W0 -= W0 % 2
        fr = fr[:H0, :W0]
        if fr.shape[0] < H0 or fr.shape[1] < W0:
            pad = np.zeros((H0, W0, 3), dtype=np.uint8)
            pad[: fr.shape[0], : fr.shape[1]] = fr
            fr = pad
        frames.append(fr)
        if progress_fn is not None:
            progress_fn(idx + 1, len(episode.steps))

    imageio.mimsave(str(out_path), frames, fps=int(fps), macro_block_size=1)
    return (W0, H0) if H0 is not None else (0, 0)


# ---------------------------------------------------------------------------
# Summary PNG (final-state snapshot)
# ---------------------------------------------------------------------------


def save_summary_figure(episode, scenario, cam_cfg, ego_cfg, out_path, dpi=110):
    """Write one PNG showing the final state of an episode."""
    if not episode.steps:
        return
    final = episode.steps[-1]
    past = [sr.pos for sr in episode.steps]
    fig = render_episode_frame(
        scenario, final, past, cam_cfg, ego_cfg, outcome=episode.outcome
    )
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
