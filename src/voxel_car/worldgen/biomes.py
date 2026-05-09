"""Heightmap → multi-material voxel grid (the 'biome' pass).

Per-column rules
----------------
For each (x, y) column with surface height ``h``:

Land columns (h >= sea_level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    z = 0                          STONE   (bedrock — always)
    0 < z < h - dirt_depth         STONE
    h - dirt_depth <= z < h        DIRT
    z == h  (and h > 0)            surface biome:
        SAND    if (h - sea_level) <= shoreline_thickness
                AND sea_level > 0     (beach)
        STONE   if h >= stone_line    (rocky peak)
        GRASS   otherwise
    z > h                          AIR

Underwater columns (h < sea_level) — only generated when sea_level > 0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    z = 0                          STONE
    0 < z < h                      STONE
    z == h                         SAND    (sea floor)
    h < z <= sea_level             WATER
    z > sea_level                  AIR

The implementation is fully vectorised over the (X, Y, Z) grid.

Note on h=0 columns
-------------------
A flat plain has h=0. We don't paint a "surface biome" voxel for h=0
columns (z==h would be the bedrock voxel at z=0, which we always force
to STONE). This is deliberate — a flat plain renders as STONE bedrock
unless something else (a road, a tree's roots, etc.) overrides it.
"""

from __future__ import annotations

import numpy as np

from ..common.materials import Material


def assign_materials(
    heights,
    grid_z,
    sea_level=0,
    dirt_depth=2,
    shoreline_thickness=1,
    stone_line=None,
):
    """Build the (X, Y, Z) uint8 material grid from a heightmap.

    Parameters
    ----------
    heights : (X, Y) int32, top voxel index per column.
    grid_z  : int, Z extent of the world grid.
    sea_level : int. 0 disables water entirely.
    dirt_depth : thickness of DIRT layer under GRASS surfaces.
    shoreline_thickness : voxels above sea_level that still render as SAND.
    stone_line : columns at or above this height get STONE on top
        (rocky peaks). ``None`` disables stone peaks.
    """
    gx, gy = heights.shape
    gz = int(grid_z)
    sea_level = int(sea_level)
    dirt_depth = int(dirt_depth)
    shoreline_thickness = int(shoreline_thickness)
    if stone_line is None:
        stone_line = gz + 1  # effectively disables
    else:
        stone_line = int(stone_line)

    materials = np.full((gx, gy, gz), int(Material.AIR), dtype=np.uint8)

    Z = np.arange(gz, dtype=np.int32)[None, None, :]  # (1, 1, Z)
    H = heights[:, :, None].astype(np.int32)  # (X, Y, 1)

    land = H >= sea_level
    underwater = ~land

    # ---- Land columns ----
    stone_sub = land & (Z < (H - dirt_depth)) & (Z > 0)
    dirt_sub = land & (Z >= (H - dirt_depth)) & (Z < H) & (Z > 0)
    materials[stone_sub] = int(Material.STONE)
    materials[dirt_sub] = int(Material.DIRT)

    # Surface row only for h > 0 columns. h=0 columns get only bedrock at z=0.
    surface = land & (Z == H) & (H > 0)
    is_shore = (sea_level > 0) & ((H - sea_level) <= shoreline_thickness)
    is_high = H >= stone_line
    materials[surface & is_shore & ~is_high] = int(Material.SAND)
    materials[surface & is_high & ~is_shore] = int(Material.STONE)
    materials[surface & ~is_shore & ~is_high] = int(Material.GRASS)

    # ---- Underwater columns (only meaningful when sea_level > 0) ----
    if sea_level > 0:
        # Sub-floor stone for h>0 underwater columns.
        uw_stone = underwater & (Z < H) & (Z > 0)
        uw_floor = underwater & (Z == H) & (H > 0)
        uw_water = underwater & (Z > H) & (Z <= sea_level)
        materials[uw_stone] = int(Material.STONE)
        materials[uw_floor] = int(Material.SAND)
        materials[uw_water] = int(Material.WATER)
        # h=0 underwater columns: water from z=1..sea_level.
        h0_uw = underwater & (H == 0)
        h0_uw_water = h0_uw & (Z > 0) & (Z <= sea_level)
        materials[h0_uw_water] = int(Material.WATER)

    # Bedrock at z=0 always STONE — this is what the car drives on for
    # flat plains and what the existing collision check skips over.
    materials[:, :, 0] = int(Material.STONE)
    return materials


def effective_surface_z(materials):
    """Return (X, Y) z-index of the topmost non-AIR voxel in each column.

    Useful for placing trees / paint and for visualisation. For columns
    that are entirely AIR (shouldn't happen with our pipeline), returns 0.
    """
    is_solid_grid = materials != int(Material.AIR)
    Z = is_solid_grid.shape[-1]
    rev = is_solid_grid[:, :, ::-1]
    first_from_top = np.argmax(rev, axis=-1)
    has_any = is_solid_grid.any(axis=-1)
    return np.where(has_any, Z - 1 - first_from_top, 0).astype(np.int32)


def surface_material(materials):
    """Material id of the topmost non-AIR voxel of each column."""
    sz = effective_surface_z(materials)
    X, Y, _ = materials.shape
    xs = np.arange(X)[:, None]
    ys = np.arange(Y)[None, :]
    return materials[xs, ys, sz]
