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
from .noise import value_noise_2d

__all__ = [
    "NUM_CAMERAS",
    "CAMERA_COLORS",
    "DEFAULT_CAMERA_SPECS",
    "CameraConfig",
    "WorldConfig",
    "TrajectoryConfig",
    "RenderConfig",
    "default_cameras",
    "value_noise_2d",
]
