"""Extended Kalman filter utilities for 2D vehicle pose estimation.

State convention
----------------
    x = [world_x, world_y, theta, v]

where:
    theta : world-frame heading angle in radians
    v     : forward speed

Control convention
------------------
    u = [yaw_rate, acceleration]

Measurement convention
----------------------
    PoseMeasurement may contain:
        xy      : measured world position, shape (2,)
        theta   : optional measured heading angle in radians
        speed   : optional measured forward speed
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass
class EKFConfig:
    """Noise and timestep configuration for VehicleEKF.

    All *_std fields are standard deviations. Heading/process angular values are
    represented in degrees here because that is easier to expose in a UI.
    """

    dt: float = 0.1

    # Process noise standard deviations.
    process_xy_std: float = 0.03
    process_theta_std_deg: float = 1.0
    process_v_std: float = 0.10

    # Measurement noise standard deviations.
    gps_xy_std: float = 1.0
    heading_std_deg: float = 8.0
    speed_std: float = 0.25


@dataclass
class PoseMeasurement:
    """A possibly-partial vehicle pose measurement.

    Parameters
    ----------
    xy:
        Position measurement [world_x, world_y].
    theta:
        Optional heading measurement in radians.
    speed:
        Optional speed measurement.
    """

    xy: np.ndarray
    theta: Optional[float] = None
    speed: Optional[float] = None


class VehicleEKF:
    """Extended Kalman filter for state [x, y, theta, v].

    The process model is a simple unicycle-style kinematic model:

        x'     = x + v dt cos(theta)
        y'     = y + v dt sin(theta)
        theta' = theta + yaw_rate dt
        v'     = v + acceleration dt

    The measurement model is linear for any subset of:
        x, y, theta, v
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray, cfg: EKFConfig):
        self.x = np.asarray(x0, dtype=np.float64).reshape(4).copy()
        self.P = np.asarray(P0, dtype=np.float64).reshape(4, 4).copy()
        self.cfg = cfg
        self.x[2] = wrap_angle(float(self.x[2]))

    def predict(
        self,
        yaw_rate: float,
        acceleration: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run EKF prediction step."""
        dt = float(self.cfg.dt)

        px, py, theta, v = [float(a) for a in self.x]
        yaw_rate = float(yaw_rate)
        acceleration = float(acceleration)

        c = math.cos(theta)
        s = math.sin(theta)

        self.x = np.array(
            [
                px + v * dt * c,
                py + v * dt * s,
                wrap_angle(theta + yaw_rate * dt),
                v + acceleration * dt,
            ],
            dtype=np.float64,
        )

        # Jacobian df/dx.
        F = np.array(
            [
                [1.0, 0.0, -v * dt * s, dt * c],
                [0.0, 1.0, v * dt * c, dt * s],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        q_xy = float(self.cfg.process_xy_std) ** 2
        q_theta = math.radians(float(self.cfg.process_theta_std_deg)) ** 2
        q_v = float(self.cfg.process_v_std) ** 2
        Q = np.diag([q_xy, q_xy, q_theta, q_v]).astype(np.float64)

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self.x[2] = wrap_angle(float(self.x[2]))

        return self.x.copy(), self.P.copy()

    def update(self, measurement: PoseMeasurement) -> tuple[np.ndarray, np.ndarray]:
        """Update from a partial pose measurement."""
        rows = []
        z_values = []
        r_values = []
        angle_row_index: Optional[int] = None

        xy = np.asarray(measurement.xy, dtype=np.float64).reshape(2)

        rows.append([1.0, 0.0, 0.0, 0.0])
        z_values.append(float(xy[0]))
        r_values.append(float(self.cfg.gps_xy_std) ** 2)

        rows.append([0.0, 1.0, 0.0, 0.0])
        z_values.append(float(xy[1]))
        r_values.append(float(self.cfg.gps_xy_std) ** 2)

        if measurement.theta is not None:
            angle_row_index = len(rows)
            rows.append([0.0, 0.0, 1.0, 0.0])
            z_values.append(wrap_angle(float(measurement.theta)))
            r_values.append(math.radians(float(self.cfg.heading_std_deg)) ** 2)

        if measurement.speed is not None:
            rows.append([0.0, 0.0, 0.0, 1.0])
            z_values.append(float(measurement.speed))
            r_values.append(float(self.cfg.speed_std) ** 2)

        H = np.asarray(rows, dtype=np.float64)
        z = np.asarray(z_values, dtype=np.float64)
        R = np.diag(r_values).astype(np.float64)

        return self._linear_update(z, H, R, angle_residual_index=angle_row_index)

    def update_position(
        self,
        measured_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience position-only update."""
        return self.update(PoseMeasurement(xy=np.asarray(measured_xy), theta=None))

    def update_position_heading(
        self,
        measured_xy: np.ndarray,
        measured_theta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience position + heading update."""
        return self.update(
            PoseMeasurement(
                xy=np.asarray(measured_xy),
                theta=float(measured_theta),
            )
        )

    def _linear_update(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        angle_residual_index: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shared linear Kalman measurement update."""
        h = H @ self.x
        y = z - h

        if angle_residual_index is not None:
            y[int(angle_residual_index)] = wrap_angle(
                float(y[int(angle_residual_index)])
            )

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[2] = wrap_angle(float(self.x[2]))

        # Joseph form, numerically safer than P = (I - KH)P.
        I = np.eye(4, dtype=np.float64)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return self.x.copy(), self.P.copy()
