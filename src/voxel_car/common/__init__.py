"""Shared configuration, constants, and lightweight utilities."""

from .config import (
    NUM_CAMERAS,
    CAMERA_COLORS,
    DEFAULT_CAMERA_SPECS,
    CameraConfig,
    WorldConfig,
    TrajectoryConfig,
    RenderConfig,
    default_cameras,
)
from .materials import (
    Material,
    MATERIAL_PALETTE,
    MATERIAL_TABLE,
    NUM_MATERIALS,
    FIRST_SOLID_ID,
    IS_SOLID,
    BLOCKS_RAY,
    IS_DRIVABLE,
    material_name,
)
from .noise import value_noise_2d

__all__ = [
    # config
    "NUM_CAMERAS",
    "CAMERA_COLORS",
    "DEFAULT_CAMERA_SPECS",
    "CameraConfig",
    "WorldConfig",
    "TrajectoryConfig",
    "RenderConfig",
    "default_cameras",
    # materials
    "Material",
    "MATERIAL_PALETTE",
    "MATERIAL_TABLE",
    "NUM_MATERIALS",
    "FIRST_SOLID_ID",
    "IS_SOLID",
    "BLOCKS_RAY",
    "IS_DRIVABLE",
    "material_name",
    # noise
    "value_noise_2d",
]
