"""Closed-loop planner on ego-frame occupancy predictions.

All knowledge of the world is indirect — the planner sees only what the
model outputs from the forward camera. At each step we:

  1. Collapse the predicted ``(Dx, Dy, Dz)`` occupancy to a 2D BEV cost
     map (``predict_to_bev_cost``). Above-ground voxels only.
  2. Run A* from the car's ego-origin to the goal cell, avoiding inflated
     obstacles (``bev_astar``).
  3. Pick a lookahead waypoint, compute desired heading, rate-limit the
     turn, and return the commanded motion (``plan_next_step``).

Fallback behaviour when A* fails keeps the sim from locking up on a single
bad frame — the car crawls forward slowly and re-plans on the next frame.
"""

from __future__ import annotations
import heapq
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np

# ---------------------------------------------------------------------------
# World <-> ego XY (2D versions of the transforms already used elsewhere)
# ---------------------------------------------------------------------------


def world_to_ego_xy(world_xy, car_pos_xy, car_heading_xy):
    """Project a world (x, y) point into ego frame (ego_x=forward, ego_y=right)."""
    hx = float(car_heading_xy[0])
    hy = float(car_heading_xy[1])
    n = math.sqrt(hx * hx + hy * hy) + 1e-12
    hx /= n
    hy /= n
    dx = float(world_xy[0]) - float(car_pos_xy[0])
    dy = float(world_xy[1]) - float(car_pos_xy[1])
    ex = dx * hx + dy * hy  # forward component
    ey = dx * hy - dy * hx  # right component  (right = (hy, -hx))
    return float(ex), float(ey)


def ego_to_world_xy(ego_xy, car_pos_xy, car_heading_xy):
    """Inverse of :func:`world_to_ego_xy`."""
    hx = float(car_heading_xy[0])
    hy = float(car_heading_xy[1])
    n = math.sqrt(hx * hx + hy * hy) + 1e-12
    hx /= n
    hy /= n
    ex = float(ego_xy[0])
    ey = float(ego_xy[1])
    wx = float(car_pos_xy[0]) + ex * hx + ey * hy
    wy = float(car_pos_xy[1]) + ex * hy - ey * hx
    return float(wx), float(wy)


def ego_xy_to_cell(ex, ey, ego_cfg):
    """Floor-style mapping from ego metric → (i, j) BEV cell index.

    Cell ``i`` covers ego_x ∈ [i·res, (i+1)·res); cell ``j`` covers ego_y ∈
    [(j - Dy/2)·res, (j+1 - Dy/2)·res). Returned indices are clipped to grid.
    """
    res = float(ego_cfg.resolution)
    i = int(math.floor(float(ex) / res))
    j = int(math.floor(float(ey) / res + ego_cfg.d_y / 2.0))
    i = max(0, min(ego_cfg.d_x - 1, i))
    j = max(0, min(ego_cfg.d_y - 1, j))
    return i, j


def cell_to_ego_xy(i, j, ego_cfg):
    """Cell indices → ego-frame metric centre."""
    res = float(ego_cfg.resolution)
    ex = (float(i) + 0.5) * res
    ey = (float(j) + 0.5 - ego_cfg.d_y / 2.0) * res
    return ex, ey


# ---------------------------------------------------------------------------
# BEV cost map
# ---------------------------------------------------------------------------


def _dilate_bool(mask, iters=1):
    """In-place-friendly 4-connected binary dilation on a 2-D bool array."""
    m = mask.copy()
    for _ in range(int(iters)):
        out = m.copy()
        out[1:, :] |= m[:-1, :]
        out[:-1, :] |= m[1:, :]
        out[:, 1:] |= m[:, :-1]
        out[:, :-1] |= m[:, 1:]
        m = out
    return m


def predict_to_bev_cost(pred_occ, fov_mask=None, z_start=1, inflate=1):
    """Collapse a predicted ``(Dx, Dy, Dz)`` occupancy into a 2-D BEV obstacle
    mask suitable for A*.

    * ``z_start`` : only voxels at height ``k ≥ z_start`` are counted.
      Defaults to 1 — excludes the always-true ground layer. Raise to 2 to
      ignore shin-height obstacles too.
    * ``fov_mask`` : optional ``(Dx, Dy, Dz)`` bool. If given, out-of-FOV
      predictions are zeroed out (they are unsupervised noise).
    * ``inflate`` : morphological dilation radius, in cells, for car width.

    Returns a ``(Dx, Dy)`` bool array — True = obstacle.
    """
    pred = np.asarray(pred_occ, dtype=bool)
    Dx, Dy, Dz = pred.shape
    if z_start >= Dz:
        return np.zeros((Dx, Dy), dtype=bool)
    upper = pred[:, :, int(z_start) :]
    if fov_mask is not None:
        mask_up = np.asarray(fov_mask, dtype=bool)[:, :, int(z_start) :]
        upper = upper & mask_up
    obst = upper.any(axis=-1)
    if inflate > 0:
        obst = _dilate_bool(obst, iters=int(inflate))
    return obst


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------

_ASTAR_MOVES_8 = [
    (-1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)),
    (1, 0, 1.0),
    (1, 1, math.sqrt(2.0)),
]


def _nearest_free(obst, cell, max_radius=6):
    """Expanding-ring search for the nearest free cell at or near ``cell``.

    Used when the naive goal-cell clipping lands on an obstacle — A* can't
    terminate on a blocked cell, so we retarget to the closest free one.
    """
    Dx, Dy = obst.shape
    i0, j0 = int(cell[0]), int(cell[1])
    if 0 <= i0 < Dx and 0 <= j0 < Dy and not obst[i0, j0]:
        return (i0, j0)
    for r in range(1, int(max_radius) + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if abs(di) != r and abs(dj) != r:
                    continue
                ni, nj = i0 + di, j0 + dj
                if 0 <= ni < Dx and 0 <= nj < Dy and not obst[ni, nj]:
                    return (ni, nj)
    return None


def bev_astar(obst, start, goal, max_expansions=10000):
    """8-connected A* on a 2-D bool obstacle grid.

    Parameters
    ----------
    obst     : (Dx, Dy) bool — True = impassable
    start    : (i, j) integer cell.
    goal     : (i, j) integer cell.

    Returns a list of ``(i, j)`` cells from start to goal inclusive, or
    ``None`` if no path exists.
    """
    Dx, Dy = obst.shape
    si, sj = int(start[0]), int(start[1])
    gi, gj = int(goal[0]), int(goal[1])
    if not (0 <= si < Dx and 0 <= sj < Dy):
        return None
    if not (0 <= gi < Dx and 0 <= gj < Dy):
        return None

    def heur(i, j):
        di, dj = i - gi, j - gj
        return math.hypot(di, dj)

    open_h = [(heur(si, sj), 0.0, si, sj)]
    best_g = {(si, sj): 0.0}
    came = {}
    expansions = 0

    while open_h and expansions < int(max_expansions):
        f, gc, ci, cj = heapq.heappop(open_h)
        if ci == gi and cj == gj:
            path = [(ci, cj)]
            while (ci, cj) in came:
                ci, cj = came[(ci, cj)]
                path.append((ci, cj))
            return list(reversed(path))
        if gc > best_g.get((ci, cj), math.inf):
            continue
        expansions += 1
        for di, dj, cst in _ASTAR_MOVES_8:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < Dx and 0 <= nj < Dy):
                continue
            if obst[ni, nj]:
                continue
            # Disallow corner-cutting through two adjacent obstacles.
            if abs(di) + abs(dj) == 2:
                if obst[ci + di, cj] and obst[ci, cj + dj]:
                    continue
            tg = gc + cst
            if tg < best_g.get((ni, nj), math.inf):
                best_g[(ni, nj)] = tg
                came[(ni, nj)] = (ci, cj)
                heapq.heappush(open_h, (tg + heur(ni, nj), tg, ni, nj))
    return None


# ---------------------------------------------------------------------------
# Planner entry
# ---------------------------------------------------------------------------


@dataclass
class PlannerResult:
    target_heading_xy: Tuple[float, float]
    step_size: float
    plan_cells: List[Tuple[int, int]] = field(default_factory=list)
    ego_goal_cell: Tuple[int, int] = (0, 0)
    cost_map: Optional[np.ndarray] = None
    status: str = "ok"  # ok | no_path | crawl


def _clip_goal_to_grid(ego_goal_xy, ego_cfg):
    """If goal is outside the grid, project along the ray from origin onto the
    grid edge. Keeps the goal direction intact so A* routes the right way."""
    ex, ey = float(ego_goal_xy[0]), float(ego_goal_xy[1])
    res = float(ego_cfg.resolution)
    max_x = (ego_cfg.d_x - 0.01) * res
    max_y = (ego_cfg.d_y / 2.0 - 0.01) * res
    min_x = 0.0
    min_y = -max_y

    # If goal is behind, clamp to a tiny forward offset; behavior handled
    # downstream (planner will aim forward and the FOV covers forward).
    if ex < 0.5 * res:
        ex = 0.5 * res

    # Scale ray (0,0)→(ex,ey) so it lands on grid edge if outside.
    t = 1.0
    if ex > max_x:
        t = min(t, max_x / ex)
    if ey > max_y and ey != 0:
        t = min(t, max_y / ey)
    if ey < min_y and ey != 0:
        t = min(t, min_y / ey)
    return ex * t, ey * t


def plan_next_step(
    pred_occ,
    ego_cfg,
    car_pos,
    car_heading,
    world_goal,
    fov_mask=None,
    step_size=1.0,
    max_turn_deg=15.0,
    lookahead_cells=3,
    z_obstacle_start=1,
    inflate=1,
    crawl_fraction=0.3,
):
    """Plan the next commanded heading & step size.

    Returns a :class:`PlannerResult`. The commanded heading is in world
    coordinates; the caller integrates by
    ``new_pos = pos + step_size · new_heading``.
    """
    Dx, Dy, Dz = pred_occ.shape
    start_cell = (0, Dy // 2)

    # --- cost map (with and without inflation) ---
    obst_hi = predict_to_bev_cost(
        pred_occ, fov_mask=fov_mask, z_start=z_obstacle_start, inflate=inflate
    )
    obst_lo = predict_to_bev_cost(
        pred_occ, fov_mask=fov_mask, z_start=z_obstacle_start, inflate=0
    )

    # Never treat the car's own cell as blocked.
    for o in (obst_hi, obst_lo):
        o[start_cell[0], start_cell[1]] = False

    # --- goal cell (clip if outside grid, retarget if on obstacle) ---
    gex, gey = world_to_ego_xy(world_goal, car_pos, car_heading)
    gex_c, gey_c = _clip_goal_to_grid((gex, gey), ego_cfg)
    gi, gj = ego_xy_to_cell(gex_c, gey_c, ego_cfg)

    # Try to land the goal on a free cell (prefer the inflated map).
    goal_cell = _nearest_free(obst_hi, (gi, gj), max_radius=6)
    if goal_cell is None:
        goal_cell = _nearest_free(obst_lo, (gi, gj), max_radius=6)
    if goal_cell is None:
        goal_cell = (gi, gj)  # fall back; A* will likely fail → crawl

    # --- A* with inflation, then without ---
    path = bev_astar(obst_hi, start_cell, goal_cell)
    status = "ok"
    used_obst = obst_hi
    if path is None:
        path = bev_astar(obst_lo, start_cell, goal_cell)
        used_obst = obst_lo
    if path is None:
        # Total failure — crawl forward with current heading, no turn.
        return PlannerResult(
            target_heading_xy=(float(car_heading[0]), float(car_heading[1])),
            step_size=float(step_size) * float(crawl_fraction),
            plan_cells=[],
            ego_goal_cell=(gi, gj),
            cost_map=obst_hi,
            status="no_path",
        )

    # --- look ahead along the path for a heading waypoint ---
    if len(path) > int(lookahead_cells):
        wp_cell = path[int(lookahead_cells)]
    elif len(path) >= 2:
        wp_cell = path[-1]
    else:
        wp_cell = path[0]

    wp_ex, wp_ey = cell_to_ego_xy(wp_cell[0], wp_cell[1], ego_cfg)
    wp_wx, wp_wy = ego_to_world_xy((wp_ex, wp_ey), car_pos, car_heading)

    dir_x = wp_wx - float(car_pos[0])
    dir_y = wp_wy - float(car_pos[1])
    dn = math.hypot(dir_x, dir_y) + 1e-12
    desired = (dir_x / dn, dir_y / dn)

    # --- rate-limit the heading change ---
    cx, cy = float(car_heading[0]), float(car_heading[1])
    cn = math.hypot(cx, cy) + 1e-12
    cx /= cn
    cy /= cn
    cos_d = cx * desired[0] + cy * desired[1]
    sin_d = cx * desired[1] - cy * desired[0]  # z-cross
    angle = math.atan2(sin_d, cos_d)
    max_turn = math.radians(float(max_turn_deg))
    if angle > max_turn:
        angle = max_turn
    if angle < -max_turn:
        angle = -max_turn
    c, s = math.cos(angle), math.sin(angle)
    new_heading = (cx * c - cy * s, cx * s + cy * c)

    return PlannerResult(
        target_heading_xy=new_heading,
        step_size=float(step_size),
        plan_cells=path,
        ego_goal_cell=(gi, gj),
        cost_map=used_obst,
        status=status,
    )


# ---------------------------------------------------------------------------
# Ground-truth collision check (only the simulator uses this — NOT the planner)
# ---------------------------------------------------------------------------


def check_collision(
    world_voxels, car_pos, car_heading, length_m=4.0, width_m=2.4, z_ignore=0
):
    """True if the car footprint overlaps any occupied voxel above ``z_ignore``.

    Ground layer is always True in the simulator's voxel grid; we skip it by
    default. This is used by the simulator to score an episode, NOT visible
    to the planner or model.
    """
    hx, hy = float(car_heading[0]), float(car_heading[1])
    n = math.hypot(hx, hy) + 1e-12
    hx /= n
    hy /= n
    rx, ry = hy, -hx
    cx, cy = float(car_pos[0]), float(car_pos[1])
    hL, hW = 0.5 * float(length_m), 0.5 * float(width_m)
    VX, VY, VZ = world_voxels.shape

    # 5-point footprint: centre + 4 corners.
    offsets = [(0.0, 0.0), (hL, hW), (hL, -hW), (-hL, hW), (-hL, -hW)]
    for du, dv in offsets:
        px = cx + du * hx + dv * rx
        py = cy + du * hy + dv * ry
        ix = int(math.floor(px))
        iy = int(math.floor(py))
        if not (0 <= ix < VX and 0 <= iy < VY):
            return True
        # Any non-ground voxel occupied in this column → collision.
        if world_voxels[ix, iy, int(z_ignore) + 1 :].any():
            return True
    return False
