"""Carve a road onto the material grid along the car's trajectory.

This is the **critical adaptation** vs voxel_drone, which carves random
edge-to-edge roads. For autonomous driving the road must be the
trajectory the planner / RL policy is going to follow, so we drive the
carve directly off the supplied waypoints.

Behavior
--------
For every cell within ``road_width`` of any waypoint:
    * all voxels above z=0 are cleared to AIR (no terrain, no water,
      no trees stay on the road),
    * the z=0 voxel is painted ROAD,
    * the heightmap is set to 0 at that cell.

For cells in the shoulder annulus (``road_width < d <= road_width +
shoulder_extra``):
    * the heightmap is faded linearly from 0 at the inner edge to its
      original value at the outer edge,
    * voxels above the new (lower) height are cleared to AIR,
    * if the cell's previous surface biome is now in mid-air, we paint
      the new top voxel as DIRT to give a "roadside cutting" look
      rather than a floating grass strip.

This carve runs *after* biome assignment and *before* tree placement,
so trees can use ``road_mask`` to keep their distance.
"""

from __future__ import annotations

import numpy as np

from ..common.materials import Material


def carve_road(
    materials,
    heights,
    trajectory,
    road_width=3,
    shoulder_extra=2,
):
    """Carve a road in place. Returns a (X, Y) bool mask of strict road cells.

    Parameters
    ----------
    materials : (X, Y, Z) uint8 — modified in place.
    heights   : (X, Y) int32 — modified in place (post-fade heights).
    trajectory : (N, 2) float — waypoints in voxel units.
    road_width : strict road radius in cells. Total drivable corridor
                 width is 2*road_width + 1.
    shoulder_extra : additional radius for the height-fading shoulder.
    """
    gx, gy, gz = materials.shape
    r = int(road_width)
    R_sh = r + int(max(0, shoulder_extra))

    road_mask = np.zeros((gx, gy), dtype=bool)
    air = int(Material.AIR)
    road = int(Material.ROAD)

    for wp in trajectory:
        ix0 = int(round(float(wp[0])))
        iy0 = int(round(float(wp[1])))
        xi0 = max(0, ix0 - R_sh)
        xi1 = min(gx, ix0 + R_sh + 1)
        yi0 = max(0, iy0 - R_sh)
        yi1 = min(gy, iy0 + R_sh + 1)
        if xi0 >= xi1 or yi0 >= yi1:
            continue

        dxg, dyg = np.meshgrid(
            np.arange(xi0, xi1) - ix0,
            np.arange(yi0, yi1) - iy0,
            indexing="ij",
        )
        d2 = (dxg * dxg + dyg * dyg).astype(np.int32)

        inside_road = d2 <= r * r
        in_shoulder = (~inside_road) & (d2 <= R_sh * R_sh)

        # ---- Strict road: clear above z=0, place ROAD at z=0 ----
        if inside_road.any():
            ix_arr, iy_arr = np.where(inside_road)
            for px, py in zip(ix_arr, iy_arr):
                ix = xi0 + int(px)
                iy = yi0 + int(py)
                materials[ix, iy, 1:] = air
                materials[ix, iy, 0] = road
                heights[ix, iy] = 0
                road_mask[ix, iy] = True

        # ---- Shoulder fade ----
        if R_sh > r and in_shoulder.any():
            ix_arr, iy_arr = np.where(in_shoulder)
            for px, py in zip(ix_arr, iy_arr):
                ix = xi0 + int(px)
                iy = yi0 + int(py)
                d = float(np.sqrt(d2[int(px), int(py)]))
                fade = (d - r) / float(R_sh - r)  # 0 at inner edge → 1 at outer
                old_h = int(heights[ix, iy])
                new_h = int(np.floor(fade * old_h))
                if new_h < old_h:
                    materials[ix, iy, new_h + 1 :] = air
                    heights[ix, iy] = new_h
                    if new_h >= 1 and materials[ix, iy, new_h] == air:
                        # We cut through the surface biome — repaint the
                        # exposed top as DIRT for a roadside-cutting look.
                        materials[ix, iy, new_h] = int(Material.DIRT)

    return road_mask
