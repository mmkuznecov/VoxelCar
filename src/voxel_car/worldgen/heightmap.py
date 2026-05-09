"""Terrain heightmap generation for the voxel-car worldgen pipeline.

Produces a per-column integer height ``heights[x, y]`` representing the
top voxel index of solid ground at that column. Heights are clamped so
the bottommost voxel (z=0) is always solid bedrock and the topmost
column never reaches the grid ceiling (we leave at least one AIR voxel
above the highest mountain so the renderer always has sky).

This is the same flat-plains-and-hills heightmap the legacy
``build_world`` produced — just factored out so the biome assignment
pass can be written cleanly against it.
"""

from __future__ import annotations

import numpy as np

from ..common.noise import value_noise_2d


def generate_heightmap(
    grid_x,
    grid_y,
    grid_z,
    seed,
    noise_scale=18.0,
    octaves=4,
    terrain_power=1.8,
    flat_threshold=0.45,
    max_obstacle_height=14,
):
    """Generate a (grid_x, grid_y) integer heightmap.

    Pipeline:
      1. Multi-octave value noise → ``hm`` ∈ [0, 1].
      2. ``shaped = hm ** terrain_power``  (>1 sharpens peaks / flattens
         valleys → more flat ground for the car to drive on).
      3. Scale to integer voxel heights in
         ``[0, max_obstacle_height]``.
      4. Force columns where ``hm < flat_threshold`` to height 0
         (this matches the legacy generator's plain-flattening rule).
      5. Cap at ``grid_z - 2`` so at least one AIR voxel remains above
         the highest column.

    Returns
    -------
    heights : (grid_x, grid_y) int32
    hm      : (grid_x, grid_y) float32 — raw noise values in [0, 1],
              useful if downstream code wants to threshold on them.
    """
    H, W = int(grid_x), int(grid_y)

    hm = value_noise_2d(
        (H, W),
        scale=float(noise_scale),
        octaves=int(octaves),
        persistence=0.5,
        seed=int(seed),
    )
    shaped = np.power(hm, float(terrain_power))

    ceil = int(min(int(max_obstacle_height), int(grid_z) - 2))
    if ceil < 1:
        raise ValueError(
            f"Cannot fit terrain in grid_z={grid_z} with "
            f"max_obstacle_height={max_obstacle_height}."
        )

    heights = (shaped * float(ceil)).astype(np.int32)
    heights[hm < float(flat_threshold)] = 0
    heights = np.clip(heights, 0, ceil)
    return heights, hm
