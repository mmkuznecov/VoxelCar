"""Procedural trajectory, world, and scenario generation."""

from .trajectory import build_trajectory
from .world import build_world, build_world_from_config, get_world_and_trajectory
from .scenarios import Scenario, SCENARIO_PRESETS, generate_scenario

# New multi-biome pipeline pieces (also re-exported for users that want
# direct access without going through build_world).
from .heightmap import generate_heightmap
from .biomes import assign_materials, effective_surface_z, surface_material
from .roads import carve_road
from .trees import plant_trees

__all__ = [
    "build_trajectory",
    "build_world",
    "build_world_from_config",
    "get_world_and_trajectory",
    "Scenario",
    "SCENARIO_PRESETS",
    "generate_scenario",
    # New pipeline modules
    "generate_heightmap",
    "assign_materials",
    "effective_surface_z",
    "surface_material",
    "carve_road",
    "plant_trees",
]
