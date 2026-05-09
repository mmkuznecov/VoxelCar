"""Tree placement on the multi-material voxel grid.

Trees only spawn on GRASS surfaces (so we don't get them on road tiles,
beach sand, or stone peaks) and stay away from the road corridor by
default. Each tree is a vertical WOOD trunk topped by a roughly-spherical
LEAVES canopy.

Why bother for a car simulator
------------------------------
Trees sit on top of GRASS columns that are *already* tall enough to
collide with the car, so they don't introduce new drivable obstacles in
the strict sense. What they DO add:

  * Tall, distinct visual landmarks for the camera and OccNet to learn.
  * A more textured edge to driving corridors — the model has to ignore
    canopy clutter and focus on the road geometry.
  * Variety across runs that's stable per seed (helps with overfitting
    to the bare-noise terrain texture).

Sequential rejection placement is fast and the result is visually
indistinguishable from Poisson-disk for the densities we use.
"""

from __future__ import annotations

import numpy as np

from ..common.materials import Material


def plant_trees(
    materials,
    heights,
    road_mask=None,
    n_trees=80,
    seed=2042,
    trunk_height_min=3,
    trunk_height_max=5,
    leaf_radius=2,
    min_separation=3,
    road_buffer=2,
    max_surface_z=None,
):
    """Plant trees in place on the material grid.

    Returns a list of (x, y) positions of successfully planted trees.
    """
    rng = np.random.RandomState(int(seed))
    gx, gy, gz = materials.shape

    if max_surface_z is None:
        # Don't plant if the canopy would clip the grid ceiling.
        max_surface_z = gz - int(trunk_height_max) - int(leaf_radius) - 2
    max_surface_z = int(max_surface_z)
    if max_surface_z < 1:
        return []

    grass_id = int(Material.GRASS)

    # Eligibility mask: GRASS surface, low enough, away from roads, away
    # from the map edge.
    surface_h = heights.astype(np.int32)
    xs = np.arange(gx)[:, None]
    ys = np.arange(gy)[None, :]
    surface_mat = materials[xs, ys, surface_h]
    eligible = (
        (surface_mat == grass_id) & (surface_h > 0) & (surface_h <= max_surface_z)
    )

    # Buffer trees away from roads — dilate the road mask by ``road_buffer``
    # cells (4-connected) and exclude.
    if road_mask is not None and int(road_buffer) > 0:
        rb = int(road_buffer)
        buf = road_mask.copy()
        for _ in range(rb):
            grown = buf.copy()
            grown[1:, :] |= buf[:-1, :]
            grown[:-1, :] |= buf[1:, :]
            grown[:, 1:] |= buf[:, :-1]
            grown[:, :-1] |= buf[:, 1:]
            buf = grown
        eligible &= ~buf

    # Avoid map edges so leaf clusters don't fall off.
    edge_pad = int(leaf_radius) + 1
    eligible[:edge_pad, :] = False
    eligible[-edge_pad:, :] = False
    eligible[:, :edge_pad] = False
    eligible[:, -edge_pad:] = False

    candidates = np.argwhere(eligible)
    if len(candidates) == 0:
        return []
    rng.shuffle(candidates)

    placed: list = []
    placed_arr = np.empty((0, 2), dtype=np.int32)
    min_sep_sq = int(min_separation) ** 2

    for x, y in candidates:
        if len(placed) >= int(n_trees):
            break

        # Reject if too close to any existing tree.
        if len(placed_arr) > 0:
            d2 = ((placed_arr - np.array([x, y])) ** 2).sum(axis=1)
            if (d2 < min_sep_sq).any():
                continue

        h = int(heights[x, y])
        trunk_h = int(rng.randint(int(trunk_height_min), int(trunk_height_max) + 1))
        leaf_top_z = h + trunk_h + int(leaf_radius)
        if leaf_top_z >= gz:
            continue  # tree would clip the grid ceiling

        # Trunk
        for tz in range(h + 1, h + 1 + trunk_h):
            materials[x, y, tz] = int(Material.WOOD)

        # Leaves: vertically-squashed sphere centred at the top of the trunk.
        cz = h + trunk_h
        r = int(leaf_radius)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-1, 2):
                    nx, ny, nz = int(x) + dx, int(y) + dy, cz + dz
                    if not (0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz):
                        continue
                    d2 = dx * dx + dy * dy + (dz * 1.5) ** 2
                    if d2 > r * r + 0.5:
                        continue
                    if materials[nx, ny, nz] == int(Material.AIR):
                        materials[nx, ny, nz] = int(Material.LEAVES)

        placed.append((int(x), int(y)))
        placed_arr = np.vstack([placed_arr, [[x, y]]])

    return placed
