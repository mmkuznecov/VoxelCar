"""Multi-biome voxel world generation, shaped around a drivable trajectory.

Pipeline
--------
    1. Heightmap            (worldgen.heightmap.generate_heightmap)
    2. Material assignment  (worldgen.biomes.assign_materials)
    3. Road carving         (worldgen.roads.carve_road) along the
       supplied trajectory — flat drivable corridor regardless of biome.
    4. Tree placement       (worldgen.trees.plant_trees) on grass
       columns away from the road. Optional.

The pipeline outputs a multi-material grid as its primary artifact and a
backward-compatible bool occupancy grid (``voxels``) derived from it via
``materials != AIR``. Every existing collision / planning / perception
consumer keeps working without modification — the bool grid has the same
semantics as before:

    voxels[i, j, k] == True   iff   the car cannot occupy that voxel

WATER counts as solid for this purpose: a car cannot drive through water,
even though it's traversable for a drone. The renderer can still distinguish
WATER from earth using the material grid (passed as an optional argument).

Backward compatibility
----------------------
``build_world(...)`` keeps its old signature and returns ``(voxels, heights)``
by default. Pass ``return_materials=True`` to also receive the material
grid and tree positions for richer rendering.
"""

from __future__ import annotations

import numpy as np

from ..common.materials import Material
from .biomes import assign_materials
from .heightmap import generate_heightmap
from .roads import carve_road
from .trees import plant_trees


def build_world(
    grid_x,
    grid_y,
    grid_z,
    seed,
    noise_scale,
    max_obstacle_height,
    road_width,
    trajectory,
    shoulder_extra=2,
    # New biome / tree controls. Defaults preserve the legacy look closely
    # (no water, no trees, no stone peaks). Enable explicitly per-scenario.
    terrain_power=1.8,
    octaves=4,
    flat_threshold=0.45,
    sea_level=0,
    shoreline_thickness=1,
    stone_line_offset=None,
    dirt_depth=2,
    n_trees=0,
    tree_seed=None,
    trunk_height_min=3,
    trunk_height_max=5,
    leaf_radius=2,
    tree_min_separation=3,
    tree_road_buffer=2,
    return_materials=False,
):
    """Generate a multi-biome voxel world shaped around the trajectory.

    Returns
    -------
    voxels : (grid_x, grid_y, grid_z) bool
        Backward-compatible occupancy array — True wherever a voxel is
        non-AIR. Compatible with all existing collision / camera code.
    heights : (grid_x, grid_y) int32
        Post-carve column-top heights.
    materials : (grid_x, grid_y, grid_z) uint8
        Only when ``return_materials=True``. Use with the materials-aware
        renderer for biome-coloured camera/BEV output.
    tree_positions : list of (x, y)
        Only when ``return_materials=True``. Empty when n_trees=0.
    """
    gx, gy, gz = int(grid_x), int(grid_y), int(grid_z)

    # 1. Heightmap.
    heights, _hm = generate_heightmap(
        gx,
        gy,
        gz,
        seed=int(seed),
        noise_scale=float(noise_scale),
        octaves=int(octaves),
        terrain_power=float(terrain_power),
        flat_threshold=float(flat_threshold),
        max_obstacle_height=int(max_obstacle_height),
    )

    # 2. Materials.
    materials = assign_materials(
        heights,
        gz,
        sea_level=int(sea_level),
        dirt_depth=int(dirt_depth),
        shoreline_thickness=int(shoreline_thickness),
        stone_line=stone_line_offset,
    )

    # 3. Carve road (modifies materials & heights in place).
    road_mask = carve_road(
        materials,
        heights,
        trajectory=np.asarray(trajectory),
        road_width=int(road_width),
        shoulder_extra=int(shoulder_extra),
    )

    # 4. Trees.
    tree_positions: list = []
    if int(n_trees) > 0:
        tree_positions = plant_trees(
            materials,
            heights,
            road_mask=road_mask,
            n_trees=int(n_trees),
            seed=(int(tree_seed) if tree_seed is not None else int(seed) + 7777),
            trunk_height_min=int(trunk_height_min),
            trunk_height_max=int(trunk_height_max),
            leaf_radius=int(leaf_radius),
            min_separation=int(tree_min_separation),
            road_buffer=int(tree_road_buffer),
        )

    # 5. Bool occupancy — backward-compatible API.
    voxels = materials != int(Material.AIR)
    voxels[:, :, 0] = True  # bedrock always solid (defensive)

    if return_materials:
        return voxels, heights, materials, tree_positions
    return voxels, heights


def build_world_from_config(world_cfg, trajectory, return_materials=False):
    """Convenience wrapper that pulls every parameter from a WorldConfig.

    Forwards the new biome / tree fields if present on the config, falling
    back to legacy-look defaults otherwise. Keeps callers like
    ``datasets.dataset.generate_sample`` simple.
    """
    return build_world(
        world_cfg.grid_x,
        world_cfg.grid_y,
        world_cfg.grid_z,
        seed=world_cfg.seed,
        noise_scale=world_cfg.noise_scale,
        max_obstacle_height=world_cfg.max_obstacle_height,
        road_width=world_cfg.road_width,
        trajectory=trajectory,
        shoulder_extra=world_cfg.shoulder_extra,
        terrain_power=getattr(world_cfg, "terrain_power", 1.8),
        octaves=getattr(world_cfg, "noise_octaves", 4),
        flat_threshold=getattr(world_cfg, "flat_threshold", 0.45),
        sea_level=getattr(world_cfg, "sea_level", 0),
        shoreline_thickness=getattr(world_cfg, "shoreline_thickness", 1),
        stone_line_offset=getattr(world_cfg, "stone_line_offset", None),
        dirt_depth=getattr(world_cfg, "dirt_depth", 2),
        n_trees=getattr(world_cfg, "n_trees", 0),
        tree_seed=getattr(world_cfg, "tree_seed", None),
        trunk_height_min=getattr(world_cfg, "trunk_height_min", 3),
        trunk_height_max=getattr(world_cfg, "trunk_height_max", 5),
        leaf_radius=getattr(world_cfg, "leaf_radius", 2),
        tree_min_separation=getattr(world_cfg, "tree_min_separation", 3),
        tree_road_buffer=getattr(world_cfg, "tree_road_buffer", 2),
        return_materials=return_materials,
    )


# ---------------------------------------------------------------------------
# Cache: shared by the Gradio preview / dataset generator so dragging a
# camera slider doesn't rebuild the voxel grid. Keys include every world
# parameter that affects the output.
# ---------------------------------------------------------------------------

_CACHE: dict = {}
_CACHE_MAX = 8


def _world_cache_key(world_cfg, traj_cfg):
    return (
        int(world_cfg.seed),
        int(world_cfg.grid_size),
        int(world_cfg.max_obstacle_height),
        int(world_cfg.road_width),
        float(world_cfg.noise_scale),
        int(world_cfg.shoulder_extra),
        float(getattr(world_cfg, "terrain_power", 1.8)),
        int(getattr(world_cfg, "noise_octaves", 4)),
        float(getattr(world_cfg, "flat_threshold", 0.45)),
        int(getattr(world_cfg, "sea_level", 0)),
        int(getattr(world_cfg, "shoreline_thickness", 1)),
        getattr(world_cfg, "stone_line_offset", None),
        int(getattr(world_cfg, "dirt_depth", 2)),
        int(getattr(world_cfg, "n_trees", 0)),
        getattr(world_cfg, "tree_seed", None),
        int(getattr(world_cfg, "trunk_height_min", 3)),
        int(getattr(world_cfg, "trunk_height_max", 5)),
        int(getattr(world_cfg, "leaf_radius", 2)),
        int(getattr(world_cfg, "tree_min_separation", 3)),
        int(getattr(world_cfg, "tree_road_buffer", 2)),
        int(traj_cfg.seed),
        int(traj_cfg.n_segments),
        float(traj_cfg.noise_amplitude),
        int(traj_cfg.smoothing_window),
        int(traj_cfg.margin),
    )


def get_world_and_trajectory(world_cfg, traj_cfg, return_materials=False):
    """Return ``(voxels, trajectory, seg_types[, materials, trees])`` cached
    on every world + trajectory parameter."""
    from .trajectory import build_trajectory

    key = _world_cache_key(world_cfg, traj_cfg)
    cached_form = "rich" if return_materials else "bool"
    full_key = (cached_form,) + key

    if full_key in _CACHE:
        return _CACHE[full_key]

    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))

    traj, seg_types = build_trajectory(
        world_cfg.grid_x,
        world_cfg.grid_y,
        seed=traj_cfg.seed,
        n_segments=traj_cfg.n_segments,
        noise_amplitude=traj_cfg.noise_amplitude,
        smoothing_window=traj_cfg.smoothing_window,
        margin=traj_cfg.margin,
    )

    if return_materials:
        voxels, _heights, materials, trees = build_world_from_config(
            world_cfg, traj, return_materials=True
        )
        result = (voxels, traj, seg_types, materials, trees)
    else:
        voxels, _heights = build_world_from_config(
            world_cfg, traj, return_materials=False
        )
        result = (voxels, traj, seg_types)

    _CACHE[full_key] = result
    return result
