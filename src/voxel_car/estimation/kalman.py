"""Linear Kalman filtering utilities.

The filter is intentionally generic and NumPy-only. Project-specific vehicle
state estimation lives in ekf.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class KalmanState:
    x: np.ndarray
    P: np.ndarray


class LinearKalmanFilter:
    """Minimal linear Kalman filter.

    State:
        x : (N,) mean
        P : (N, N) covariance

    Dynamics:
        x = F x + B u
        P = F P F.T + Q

    Measurement:
        z = H x + noise
    """

    def __init__(self, x0, P0):
        self.x = np.asarray(x0, dtype=np.float64).copy()
        self.P = np.asarray(P0, dtype=np.float64).copy()

    def predict(self, F, Q, B=None, u=None):
        F = np.asarray(F, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)

        if B is not None and u is not None:
            B = np.asarray(B, dtype=np.float64)
            u = np.asarray(u, dtype=np.float64)
            self.x = F @ self.x + B @ u
        else:
            self.x = F @ self.x

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy(), self.P.copy()

    def update(self, z, H, R):
        z = np.asarray(z, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(self.P.shape[0], dtype=np.float64)
        # Joseph form is numerically safer than P = (I - KH)P.
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return self.x.copy(), self.P.copy()
