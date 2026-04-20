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


def _distance_transform(obst):
    """Cheap Chebyshev distance transform on a 2-D bool grid.

    Returns an int array where 0 = obstacle, 1 = neighbor of obstacle, etc.
    Used as a soft cost gradient around obstacles so A* prefers the middle of
    corridors rather than hugging walls.
    """
    Dx, Dy = obst.shape
    INF = Dx + Dy + 10
    d = np.where(obst, 0, INF).astype(np.int32)
    # Two-pass serial Chebyshev DT.
    for i in range(Dx):
        for j in range(Dy):
            v = d[i, j]
            if i > 0:
                v = min(v, d[i - 1, j] + 1)
            if j > 0:
                v = min(v, d[i, j - 1] + 1)
            if i > 0 and j > 0:
                v = min(v, d[i - 1, j - 1] + 1)
            if i > 0 and j < Dy - 1:
                v = min(v, d[i - 1, j + 1] + 1)
            d[i, j] = v
    for i in range(Dx - 1, -1, -1):
        for j in range(Dy - 1, -1, -1):
            v = d[i, j]
            if i < Dx - 1:
                v = min(v, d[i + 1, j] + 1)
            if j < Dy - 1:
                v = min(v, d[i, j + 1] + 1)
            if i < Dx - 1 and j < Dy - 1:
                v = min(v, d[i + 1, j + 1] + 1)
            if i < Dx - 1 and j > 0:
                v = min(v, d[i + 1, j - 1] + 1)
            d[i, j] = v
    return d


def predict_to_bev_cost(
    pred_occ,
    fov_mask=None,
    z_start=1,
    inflate=2,
    close_range_cells=3,
    treat_unknown_close_as_obstacle=True,
):
    """Collapse a predicted ``(Dx, Dy, Dz)`` occupancy into a 2-D BEV obstacle
    mask suitable for A*.

    Important subtlety — the **close-range blind spot**:
        The camera is mounted at (fwd=1.5, height=2). Ego voxels very close to
        the car — roughly ``ego_x ≤ 3m`` — fall *below* the camera's vertical
        FOV and are therefore unsupervised by training. If we naïvely AND the
        prediction with the FOV mask, those cells become "free" in our cost
        map, and A* will cheerfully route through a 2-metre wall that happens
        to sit in the blind spot. Driving, empirically, straight into said wall.

        The fix here is two-fold:
        (a) Keep the raw prediction (do NOT AND with FOV mask) for cells at
            ``ego_x < close_range_cells``. The model might still predict them
            correctly since the loss was masked, not the training-time inputs.
        (b) Optionally mark those close-range cells as *obstacles* by default
            when both the prediction and the FOV mask say "unknown" — the
            pessimistic choice stops us from driving into them.

    Parameters
    ----------
    pred_occ : (Dx, Dy, Dz) bool — model prediction.
    fov_mask : optional (Dx, Dy, Dz) bool — supervised voxels.
    z_start  : only voxels at height ``k ≥ z_start`` count as obstacles.
               Default 1 excludes the always-true ground layer.
    inflate  : morphological dilation radius in cells (car half-width safety).
               Default 2 gives a 2 m safety buffer for the 2.4 m wide car.
    close_range_cells : cells within this many of the car are handled
               pessimistically (see note above).
    treat_unknown_close_as_obstacle : if True, close-range cells that are
               neither in the FOV nor predicted occupied are still marked
               as obstacles — strongest "don't drive into blind spots" policy.

    Returns a ``(Dx, Dy)`` bool array — True = obstacle.
    """
    pred = np.asarray(pred_occ, dtype=bool)
    Dx, Dy, Dz = pred.shape
    if z_start >= Dz:
        return np.zeros((Dx, Dy), dtype=bool)
    upper = pred[:, :, int(z_start) :]  # (Dx, Dy, Dz')

    if fov_mask is not None:
        mask_up = np.asarray(fov_mask, dtype=bool)[:, :, int(z_start) :]
        # Real obstacles = predicted AND in-FOV (where the model was supervised).
        # Plus, for close-range rows, the raw prediction (no FOV gate) since
        # those rows are the blind spot where we trust the model's best guess.
        obst_in_fov = (upper & mask_up).any(axis=-1)  # (Dx, Dy)
        obst_raw = upper.any(axis=-1)
        fov_any_z = mask_up.any(axis=-1)

        real_obst = obst_in_fov.copy()
        cr = int(close_range_cells)
        if cr > 0:
            real_obst[:cr] = real_obst[:cr] | obst_raw[:cr]

        # Blind-spot pessimism (kept SEPARATE so it doesn't get dilated).
        blind_obst = np.zeros_like(real_obst)
        if treat_unknown_close_as_obstacle and cr > 0:
            blind_obst[:cr] = ~fov_any_z[:cr] & ~obst_raw[:cr]
    else:
        real_obst = upper.any(axis=-1)
        blind_obst = np.zeros_like(real_obst)
        obst_raw = real_obst  # alias

    # Dilate the REAL obstacles for the car's half-width. Don't dilate the
    # blind-spot pessimism — it's already a whole curtain of "unknown" and
    # dilating it would bleed into the legitimate forward corridor where the
    # FOV starts to cover.
    if inflate > 0:
        real_obst = _dilate_bool(real_obst, iters=int(inflate))

    obst = real_obst | blind_obst

    # AFTER dilation and pessimism: carve a narrow forward corridor through
    # the close-range cells so the car can physically advance. Only opens
    # cells that the raw prediction said were free (we never carve through a
    # predicted real obstacle).
    if fov_mask is not None and treat_unknown_close_as_obstacle:
        cr = int(close_range_cells)
        j_mid = Dy // 2
        for i in range(min(cr, Dx)):
            for dj in (-1, 0, 1):
                jj = j_mid + dj
                if 0 <= jj < Dy and not obst_raw[i, jj]:
                    obst[i, jj] = False

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


def bev_astar(
    obst, start, goal, max_expansions=10000, soft_cost=None, heading_penalty=0.0
):
    """8-connected A* on a 2-D bool obstacle grid.

    Parameters
    ----------
    obst     : (Dx, Dy) bool — True = impassable.
    start    : (i, j) integer cell.
    goal     : (i, j) integer cell.
    soft_cost : optional (Dx, Dy) float array — per-cell additional cost
        *added* to the step into that cell. Use with a distance-transform-
        based field to push A* into the middle of corridors.
    heading_penalty : optional cost added when the step direction changes
        vs. the parent's incoming direction. 0 disables. Small values like
        0.3 discourage wiggling without blocking diagonal moves outright.

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

    # State includes the direction we entered the cell from, so we can apply
    # a heading-change penalty for the next move.
    open_h = [(heur(si, sj), 0.0, si, sj, 0, 0)]
    best_g = {(si, sj, 0, 0): 0.0}
    came = {}
    expansions = 0

    while open_h and expansions < int(max_expansions):
        f, gc, ci, cj, pdi, pdj = heapq.heappop(open_h)
        if ci == gi and cj == gj:
            path = [(ci, cj)]
            key = (ci, cj, pdi, pdj)
            while key in came:
                pi, pj, ppdi, ppdj = came[key]
                path.append((pi, pj))
                key = (pi, pj, ppdi, ppdj)
            return list(reversed(path))
        if gc > best_g.get((ci, cj, pdi, pdj), math.inf):
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
            step_cost = cst
            if soft_cost is not None:
                step_cost += float(soft_cost[ni, nj])
            if heading_penalty > 0.0 and (pdi, pdj) != (0, 0):
                # Cosine angle between incoming (pdi, pdj) and (di, dj).
                a = math.hypot(pdi, pdj) + 1e-9
                b = math.hypot(di, dj) + 1e-9
                dot = (pdi * di + pdj * dj) / (a * b)
                # 1 - dot ∈ [0, 2]; straight = 0 cost, back = 2·penalty.
                step_cost += float(heading_penalty) * (1.0 - dot)
            tg = gc + step_cost
            key = (ni, nj, di, dj)
            if tg < best_g.get(key, math.inf):
                best_g[key] = tg
                came[key] = (ci, cj, pdi, pdj)
                heapq.heappush(open_h, (tg + heur(ni, nj), tg, ni, nj, di, dj))
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


def _pick_subgoal(obst, world_goal_ego, ego_cfg, start_cell, forward_bias=0.7):
    """Choose a reachable sub-goal cell inside the grid.

    Rather than always pointing A* at the (clipped) global-goal direction —
    which can force the plan toward a cell that happens to sit on the edge of
    free space and pull the car into obstacles — we look for the **furthest
    forward-reachable free cell** whose lateral offset roughly agrees with
    the goal direction. This keeps progress toward the global goal while
    prioritising visibly-free forward travel.

    Parameters
    ----------
    obst : (Dx, Dy) bool cost map (obstacles only, no inflation beyond what
           the caller already applied).
    world_goal_ego : (ex, ey) — the global goal *projected* into ego metric
           coordinates (NOT clipped to grid). Used only for direction.
    start_cell : (i, j) car cell.
    forward_bias : float in [0, 1]. 1.0 = always pick straight-ahead subgoal;
           0.0 = strictly follow the world-goal ray. Default 0.7 biases
           toward forward motion (safer when goal is far / behind obstacles).
    """
    Dx, Dy = obst.shape
    si, sj = int(start_cell[0]), int(start_cell[1])
    res = float(ego_cfg.resolution)

    # Unit vector toward goal in ego metric (ex=forward, ey=right).
    gex, gey = float(world_goal_ego[0]), float(world_goal_ego[1])
    # Force the sub-goal to be in front of the car; if goal is behind, we
    # substitute "straight ahead".
    if gex <= 0.0:
        gex = 1.0
        gey = 0.0
    gn = math.hypot(gex, gey) + 1e-12
    gdx, gdy = gex / gn, gey / gn

    # Blended direction: weighted mix of goal-direction and straight-ahead.
    bx = forward_bias * 1.0 + (1.0 - forward_bias) * gdx
    by = (1.0 - forward_bias) * gdy
    bn = math.hypot(bx, by) + 1e-12
    bx /= bn
    by /= bn

    # Scan forward from the car. For each i (forward row), find the free cell
    # in that row closest to the blended ray. Pick the furthest such i where
    # we can reach a free cell close to the desired line.
    best = None
    best_score = -1.0
    for i in range(si + 1, Dx):
        # Desired y offset from car centre, in cells, for this forward row.
        # For blended dir (bx, by) parameterised by t=i-si, the ray is
        # at (i, sj + t · by / bx).
        if abs(bx) < 1e-6:
            desired_j = sj
        else:
            t = float(i - si)
            desired_j = sj + t * (by / bx)
        # Find nearest free j in this row, within a reasonable window.
        window = int(min(Dy, 1 + i * 0.6))  # window grows with depth
        j0 = max(0, int(round(desired_j)) - window)
        j1 = min(Dy, int(round(desired_j)) + window + 1)
        row = obst[i, j0:j1]
        if row.all():
            # row fully blocked → stop the scan; no valid subgoal further out.
            break
        free_js = np.where(~row)[0]
        # Pick the one closest to desired_j.
        local_j = free_js[np.argmin(np.abs(free_js - (desired_j - j0)))]
        cand_j = int(j0 + local_j)
        # Score: depth (i) minus lateral error from the blended ray.
        score = float(i) - 0.5 * abs(cand_j - desired_j)
        if score > best_score:
            best_score = score
            best = (int(i), int(cand_j))

    if best is None:
        # Fallback — straight-ahead row by row from the car.
        for i in range(si + 1, Dx):
            if not obst[i, sj]:
                return (i, sj)
        return (si + 1 if si + 1 < Dx else si, sj)
    return best


def _mark_world_bounds_as_obstacles(
    obst, ego_cfg, car_pos, car_heading, world_shape, margin_m
):
    """For each ego cell, check if its world-frame centre falls outside the
    world (with margin) and mark it as an obstacle if so.

    This prevents the planner from routing toward the map edge when the
    camera can't provide obstacle evidence (e.g. looking at sky).
    """
    Dx, Dy = obst.shape
    WX, WY = int(world_shape[0]), int(world_shape[1])
    res = float(ego_cfg.resolution)

    hx = float(car_heading[0])
    hy = float(car_heading[1])
    n = math.sqrt(hx * hx + hy * hy) + 1e-12
    hx /= n
    hy /= n
    cx, cy = float(car_pos[0]), float(car_pos[1])
    margin = float(margin_m)

    for i in range(Dx):
        ex = (i + 0.5) * res
        for j in range(Dy):
            ey = (j + 0.5 - Dy / 2.0) * res
            wx = cx + ex * hx + ey * hy
            wy = cy + ex * hy - ey * hx
            if wx < margin or wx > WX - margin or wy < margin or wy > WY - margin:
                obst[i, j] = True


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
    inflate=2,
    close_range_cells=3,
    treat_unknown_close_as_obstacle=True,
    soft_cost_weight=4.0,
    heading_penalty=0.3,
    forward_bias=0.7,
    crawl_fraction=0.25,
    world_shape=None,
    world_margin=2.0,
    goal_slow_radius=10.0,
    goal_slow_min_fraction=0.25,
):
    """Plan the next commanded heading & step size.

    Returns a :class:`PlannerResult`. The commanded heading is in world
    coordinates; the caller integrates by
    ``new_pos = pos + step_size · new_heading``.

    Parameters worth knowing
    ------------------------
    inflate : morphological obstacle dilation (cells). 2 ≈ 2m safety buffer.
    close_range_cells : rows ``i < close_range_cells`` treated pessimistically
        (see :func:`predict_to_bev_cost`). Default 3 covers the camera blind spot.
    soft_cost_weight : multiplies a distance-transform soft cost around
        obstacles. Higher = A* hugs the middle of corridors more strongly.
    heading_penalty : small cost for heading changes, smooths the plan.
    forward_bias : 0..1 — 1.0 "always aim straight", 0.0 "always aim at goal".
        Default 0.7 keeps progress toward goal while preferring visibly-free forward travel.
    world_shape : optional (WX, WY) world dimensions. When provided, ego cells
        whose corresponding world point falls outside ``[world_margin,
        WX-world_margin]`` × likewise for Y are marked as obstacles. Prevents
        the planner from driving off the map edge when the camera sees only
        sky (a failure mode not observable from model predictions alone, since
        sky is just "empty FOV" to the model).
    world_margin : metres of buffer inside the world bounds to avoid.
    goal_slow_radius : within this distance of the world goal, step_size is
        reduced so the car doesn't overshoot.
    goal_slow_min_fraction : minimum fraction of step_size used when inside
        the goal-slow radius (at d=0 the car commits ~0 motion; at d=radius,
        full step).
    """
    Dx, Dy, Dz = pred_occ.shape
    start_cell = (0, Dy // 2)

    # --- cost maps ---
    obst_hi = predict_to_bev_cost(
        pred_occ,
        fov_mask=fov_mask,
        z_start=z_obstacle_start,
        inflate=int(inflate),
        close_range_cells=int(close_range_cells),
        treat_unknown_close_as_obstacle=bool(treat_unknown_close_as_obstacle),
    )
    obst_lo = predict_to_bev_cost(
        pred_occ,
        fov_mask=fov_mask,
        z_start=z_obstacle_start,
        inflate=0,
        close_range_cells=int(close_range_cells),
        treat_unknown_close_as_obstacle=bool(treat_unknown_close_as_obstacle),
    )

    # --- world boundary obstacles -----------------------------------------
    # When the camera sees mostly sky (near map edges, or on high ground with
    # a forward cliff), the model's prediction in the FOV is "free" — which is
    # technically correct since there's nothing to see. But "nothing visible"
    # ≠ "safe to drive forward". Add the world boundary as an obstacle in the
    # cost map so the planner refuses to aim at cells outside the world.
    if world_shape is not None:
        _mark_world_bounds_as_obstacles(
            obst_hi, ego_cfg, car_pos, car_heading, world_shape, world_margin
        )
        _mark_world_bounds_as_obstacles(
            obst_lo, ego_cfg, car_pos, car_heading, world_shape, world_margin
        )

    # Never treat the car's own cell as blocked.
    for o in (obst_hi, obst_lo):
        o[start_cell[0], start_cell[1]] = False

    # --- soft cost: distance transform of the inflated obstacle map ---
    # Cells deep inside free space get low extra cost; cells adjacent to
    # obstacles get high extra cost. This keeps A* near the middle of
    # corridors rather than hugging walls.
    dt = _distance_transform(obst_hi).astype(np.float32)
    # soft_cost = weight * exp(-dt). Adjacent to obstacle (dt=1) → cost ~0.37·w;
    # deep free (dt=5) → cost ~0.007·w (negligible).
    soft_cost = soft_cost_weight * np.exp(-dt)

    # --- goal in ego metric, no clipping yet ---
    gex, gey = world_to_ego_xy(world_goal, car_pos, car_heading)

    # --- choose sub-goal cell that is REACHABLE AND FORWARD ---
    subgoal = _pick_subgoal(
        obst_hi, (gex, gey), ego_cfg, start_cell, forward_bias=forward_bias
    )

    # --- A* on inflated cost map with soft cost + heading penalty ---
    path = bev_astar(
        obst_hi,
        start_cell,
        subgoal,
        soft_cost=soft_cost,
        heading_penalty=heading_penalty,
    )
    status = "ok"
    used_obst = obst_hi

    if path is None:
        # Retry with the non-inflated map (tight squeeze through narrow gaps).
        path = bev_astar(
            obst_lo,
            start_cell,
            subgoal,
            soft_cost=None,
            heading_penalty=heading_penalty,
        )
        used_obst = obst_lo
        status = "tight"

    if path is None:
        # Retry one more time with subgoal picked on the non-inflated map,
        # because the gap may only be reachable through cells inflation killed.
        subgoal2 = _pick_subgoal(
            obst_lo, (gex, gey), ego_cfg, start_cell, forward_bias=forward_bias
        )
        path = bev_astar(
            obst_lo,
            start_cell,
            subgoal2,
            soft_cost=None,
            heading_penalty=heading_penalty,
        )
        used_obst = obst_lo

    if path is None:
        # Total failure — crawl forward slowly. Don't change heading; the
        # next frame gives a fresh prediction.
        return PlannerResult(
            target_heading_xy=(float(car_heading[0]), float(car_heading[1])),
            step_size=float(step_size) * float(crawl_fraction),
            plan_cells=[],
            ego_goal_cell=subgoal,
            cost_map=obst_hi,
            status="no_path",
        )

    # --- look ahead along the path for a heading waypoint ---
    # Rather than picking a single cell (which is biased by cell-centre offset
    # when Dy is even: the car's ego_y=0 line sits between cells j=Dy/2-1 and
    # j=Dy/2, so cells in column Dy/2 are centred at +0.5·res), we average the
    # ego positions of a window of cells around the lookahead, then subtract
    # the cell-centre offset of the START cell so a perfectly-straight plan
    # yields exactly zero heading change.
    lh = int(lookahead_cells)
    if len(path) <= 1:
        wp_cells = [path[0]]
    else:
        lo = max(1, lh - 1)
        hi = min(len(path), lh + 3)
        if lo >= hi:
            wp_cells = [path[-1]]
        else:
            wp_cells = path[lo:hi]

    exs, eys = [], []
    for wi, wj in wp_cells:
        ex, ey = cell_to_ego_xy(wi, wj, ego_cfg)
        exs.append(ex)
        eys.append(ey)
    wp_ex = float(np.mean(exs))
    wp_ey = float(np.mean(eys))

    # Subtract the start cell's ego centre so a straight plan → zero heading.
    start_ex, start_ey = cell_to_ego_xy(start_cell[0], start_cell[1], ego_cfg)
    dex_m = wp_ex - start_ex
    dey_m = wp_ey - start_ey

    # Rotate ego-frame direction into world.
    en = math.hypot(dex_m, dey_m) + 1e-12
    dex, dey = dex_m / en, dey_m / en
    cx, cy = float(car_heading[0]), float(car_heading[1])
    cn = math.hypot(cx, cy) + 1e-12
    cx /= cn
    cy /= cn
    rx, ry = cy, -cx  # car right in world
    desired = (dex * cx + dey * rx, dex * cy + dey * ry)
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

    # --- goal-proximity slowdown ---
    # When close to the world goal, reduce step size so the car doesn't
    # overshoot past it. This is specifically to avoid the "car reaches the
    # goal radius but keeps driving forward because the prediction is clear"
    # failure: the car hits map edge 3m past the goal.
    d_goal_world = math.hypot(
        float(world_goal[0]) - float(car_pos[0]),
        float(world_goal[1]) - float(car_pos[1]),
    )
    eff_step = float(step_size)
    if d_goal_world < float(goal_slow_radius):
        frac = max(
            float(goal_slow_min_fraction), d_goal_world / float(goal_slow_radius)
        )
        eff_step = float(step_size) * frac

    return PlannerResult(
        target_heading_xy=new_heading,
        step_size=eff_step,
        plan_cells=path,
        ego_goal_cell=subgoal,
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
