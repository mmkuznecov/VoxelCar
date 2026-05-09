"""Voxel material classes, colour palette, and physical properties.

Adapted from voxel_drone for the autonomous-driving setting:

* WATER is treated as a collision obstacle for the car (a car cannot drive
  onto water tiles, unlike a drone which can fly above them). It still
  blocks rays so it renders as a blue surface.
* ROAD is a special drivable surface placed at column height 0 along the
  car's planned trajectory by ``worldgen.roads``.
* Trees (WOOD trunk + LEAVES canopy) sit on top of GRASS columns and
  therefore don't introduce new drivable obstacles for the car — the
  underlying GRASS column is already a collidable obstacle. They do add
  visual landmarks the perception model has to learn to handle.

Conventions
-----------
* Material id is a uint8 in [0, 255]. AIR = 0 so a zero-initialised grid
  is "all air".
* ``IS_SOLID``    : voxels that block the car. Used by collision checks.
* ``BLOCKS_RAY``  : voxels visible to the camera. WATER blocks rays so the
                    surface looks blue rather than transparent.
* ``IS_DRIVABLE`` : surface materials the car *should* drive on (info
                    only; useful for cost-map shaping).

Backward compatibility with the old bool API
--------------------------------------------
Any callsite that wants the legacy bool-voxel grid can use
``materials != Material.AIR``. This treats every non-AIR material —
including water — as an obstacle, which is exactly what we want for the
car. ``worldgen.world.build_world`` returns this bool grid as the first
output, so existing collision / planning / perception code keeps working
without changes.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Material(IntEnum):
    """Material classes for voxels.

    Order is significant: keep AIR=0 (so empty grids default to air).
    """

    AIR = 0
    WATER = 1
    SAND = 2
    GRASS = 3
    DIRT = 4
    STONE = 5
    WOOD = 6
    LEAVES = 7
    ROAD = 8


FIRST_SOLID_ID = 2
NUM_MATERIALS = len(Material)


# RGB palette indexed by Material id.
MATERIAL_PALETTE = np.array(
    [
        [0, 0, 0],  # AIR     (never displayed)
        [40, 95, 175],  # WATER
        [220, 200, 130],  # SAND
        [80, 150, 55],  # GRASS
        [110, 80, 50],  # DIRT
        [130, 130, 130],  # STONE
        [90, 60, 35],  # WOOD
        [55, 135, 55],  # LEAVES
        [70, 72, 78],  # ROAD
    ],
    dtype=np.uint8,
)
assert MATERIAL_PALETTE.shape == (NUM_MATERIALS, 3)


# Per-material flags. Indexed by Material id.
#   is_solid    — collides with car (only relevant at z >= 1; bedrock at z=0
#                 is always solid but check_collision skips z=0 anyway).
#   blocks_ray  — first-hit ray-march stops on this voxel.
#   is_drivable — surface material the car *should* drive on.
MATERIAL_TABLE = np.array(
    [
        # is_solid, blocks_ray, is_drivable
        [0, 0, 0],  # AIR
        [1, 1, 0],  # WATER  (impassable to the car, but visible)
        [1, 1, 0],  # SAND
        [1, 1, 0],  # GRASS
        [1, 1, 0],  # DIRT
        [1, 1, 0],  # STONE
        [1, 1, 0],  # WOOD
        [1, 1, 0],  # LEAVES
        [1, 1, 1],  # ROAD
    ],
    dtype=np.uint8,
)
assert MATERIAL_TABLE.shape == (NUM_MATERIALS, 3)


IS_SOLID = MATERIAL_TABLE[:, 0].astype(bool)
BLOCKS_RAY = MATERIAL_TABLE[:, 1].astype(bool)
IS_DRIVABLE = MATERIAL_TABLE[:, 2].astype(bool)


def material_name(mat_id) -> str:
    """Human-readable name of a material id, useful for debug printing."""
    try:
        return Material(int(mat_id)).name
    except ValueError:
        return f"UNKNOWN({int(mat_id)})"
