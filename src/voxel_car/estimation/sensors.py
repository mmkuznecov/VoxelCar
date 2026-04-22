"""Synthetic sensor helpers for state-estimation demos."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .ekf import PoseMeasurement, wrap_angle


def noisy_pose_measurement(
    true_state: np.ndarray,
    *,
    gps_xy_std: float = 1.0,
    heading_std_deg: float = 8.0,
    speed_std: float = 0.25,
    use_heading: bool = True,
    use_speed: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> PoseMeasurement:
    """Create a noisy pose measurement from a true vehicle state.

    Parameters
    ----------
    true_state:
        Array-like [x, y, theta, v].
    gps_xy_std:
        Standard deviation of x/y position noise in metres.
    heading_std_deg:
        Standard deviation of heading noise in degrees.
    speed_std:
        Standard deviation of speed noise.
    use_heading:
        Whether to include theta in the returned measurement.
    use_speed:
        Whether to include speed in the returned measurement.
    rng:
        Optional NumPy RandomState for deterministic demos.
    """
    if rng is None:
        rng = np.random.RandomState()

    state = np.asarray(true_state, dtype=np.float64).reshape(4)

    xy = state[:2] + rng.randn(2) * float(gps_xy_std)

    theta = None
    if bool(use_heading):
        theta = wrap_angle(
            float(state[2]) + rng.randn() * math.radians(float(heading_std_deg))
        )

    speed = None
    if bool(use_speed):
        speed = float(state[3]) + rng.randn() * float(speed_std)

    return PoseMeasurement(
        xy=xy.astype(np.float64),
        theta=theta,
        speed=speed,
    )
