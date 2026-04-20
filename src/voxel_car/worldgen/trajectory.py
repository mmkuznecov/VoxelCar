"""Random composite trajectories made of line / arc / sine segments.

Tangent continuity is enforced at every segment boundary:

  * ``line(L)``
        (x, y) += L · (cos θ, sin θ)            →  heading unchanged.

  * ``arc(R, L)`` — signed radius R (+ = turn left, − = turn right), arc-length L
        centre = pos + R · (-sin θ, cos θ)
        pos'   = pos + R · ( sin(θ+φ) - sin θ ,  -cos(θ+φ) + cos θ )  with φ = L/R
        θ'     = θ + φ

  * ``sine(L, amp, cycles)``
        Perpendicular offset along the segment uses an envelope
        ``env(τ) = sin²(π·τ)``   (τ = arc-length along segment / L)
        which is C¹-zero at both τ=0 and τ=1 — so the end-tangent is exactly
        the forward direction and the segment joins the next one cleanly.

After composition, a small Gaussian noise is added per-waypoint, followed by
a box-filter smoothing, and finally a hard clip to ``[margin, grid − margin]``.
"""

from __future__ import annotations
import math
import numpy as np

# ---------------------------------------------------------------------------
# Segment builders
# ---------------------------------------------------------------------------


def _line_segment(pos, theta, length, n_pts):
    """Straight line of length ``length`` along heading ``theta``.

    Returns (new_pos, new_theta, seg_Nx2). ``seg`` excludes the start point.
    """
    ts = np.linspace(0.0, float(length), int(n_pts) + 1)[1:]
    seg = np.stack(
        [pos[0] + ts * math.cos(theta), pos[1] + ts * math.sin(theta)], axis=1
    ).astype(np.float32)
    return seg[-1].copy(), float(theta), seg


def _arc_segment(pos, theta, radius_signed, length, n_pts):
    """Circular arc of signed radius (+ = left turn) and arc-length ``length``."""
    R = float(radius_signed)
    phi_total = float(length) / R
    phis = np.linspace(0.0, phi_total, int(n_pts) + 1)[1:]
    dx = R * (np.sin(theta + phis) - math.sin(theta))
    dy = R * (-np.cos(theta + phis) + math.cos(theta))
    seg = np.stack([pos[0] + dx, pos[1] + dy], axis=1).astype(np.float32)
    return seg[-1].copy(), float(theta + phi_total), seg


def _sine_segment(pos, theta, length, amplitude, cycles, n_pts):
    """Perpendicular sinusoidal wiggle with C¹-zero envelope at both ends."""
    fwd = (math.cos(theta), math.sin(theta))
    perp = (-math.sin(theta), math.cos(theta))  # left of fwd
    ts = np.linspace(0.0, float(length), int(n_pts) + 1)[1:]
    tau = ts / float(length)
    env = np.sin(math.pi * tau) ** 2  # zero value & slope at ends
    wig = float(amplitude) * env * np.sin(2.0 * math.pi * float(cycles) * tau)
    xs = pos[0] + ts * fwd[0] + wig * perp[0]
    ys = pos[1] + ts * fwd[1] + wig * perp[1]
    seg = np.stack([xs, ys], axis=1).astype(np.float32)
    # Heading at end: with the sin²(πτ) envelope, the end-tangent is exactly
    # along fwd analytically; we still recompute from the last pair as a guard.
    if len(seg) >= 2:
        d = seg[-1] - seg[-2]
        new_theta = math.atan2(float(d[1]), float(d[0]))
    else:
        new_theta = float(theta)
    return seg[-1].copy(), new_theta, seg


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _smooth_1d(x, window):
    """Box-filter smoothing that preserves the first/last (window//2) samples."""
    window = int(window)
    if window <= 1 or len(x) <= window:
        return x
    k = np.ones(window, dtype=np.float32) / window
    y = np.convolve(x, k, mode="same").astype(np.float32)
    m = window // 2
    y[:m] = x[:m]
    y[-m:] = x[-m:]
    return y


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def build_trajectory(
    grid_x,
    grid_y,
    seed,
    n_segments=6,
    noise_amplitude=0.4,
    smoothing_window=5,
    margin=5,
):
    """Compose a random drivable trajectory.

    Returns
    -------
    waypoints : (N, 2) float32 array of (x, y) positions (voxel units)
    seg_types : list[str]       per-segment type label (for metadata)
    """
    rng = np.random.RandomState(int(seed))
    gx, gy = int(grid_x), int(grid_y)
    m = float(margin)

    # Start near left edge, heading roughly into +X.
    pos = np.array(
        [m + 2.0, gy * 0.5 + rng.uniform(-gy * 0.10, gy * 0.10)], dtype=np.float32
    )
    theta = float(rng.uniform(-0.35, 0.35))

    # Nominal per-segment forward travel so the full route roughly crosses X.
    nominal = (gx - 2.0 * m) / max(int(n_segments), 1) * 1.2

    seg_types: list = []
    all_pts = [pos.copy()]

    for _ in range(int(n_segments)):
        stype = rng.choice(["line", "arc", "sine"], p=[0.35, 0.40, 0.25])
        seg_types.append(str(stype))
        L = float(rng.uniform(0.7, 1.3) * nominal)

        if stype == "line":
            n = max(4, int(round(L * 1.5)))
            pos, theta, seg = _line_segment(pos, theta, L, n)

        elif stype == "arc":
            R = float(rng.uniform(0.25, 0.60) * gx) * float(rng.choice([-1.0, 1.0]))
            # Don't let the arc sweep more than ~0.9π — avoids doubling back.
            max_phi = 0.9 * math.pi
            if abs(L / R) > max_phi:
                L = max_phi * abs(R)
            n = max(6, int(round(L * 2.0)))
            pos, theta, seg = _arc_segment(pos, theta, R, L, n)

        else:  # sine
            amp = float(rng.uniform(1.5, 4.0))
            cycles = float(rng.uniform(0.8, 2.2))
            n = max(8, int(round(L * 2.5)))
            pos, theta, seg = _sine_segment(pos, theta, L, amp, cycles, n)

        all_pts.extend(seg)

    wp = np.stack(all_pts, axis=0).astype(np.float32)

    # Per-waypoint Gaussian noise + box smoothing (keeps the path drivable but
    # removes the overly synthetic look of pure parametric curves).
    noise = rng.randn(*wp.shape).astype(np.float32) * float(noise_amplitude)
    wp_n = wp + noise
    wp_n[:, 0] = _smooth_1d(wp_n[:, 0], int(smoothing_window))
    wp_n[:, 1] = _smooth_1d(wp_n[:, 1], int(smoothing_window))

    # Clip into the grid with a safety margin.
    wp_n[:, 0] = np.clip(wp_n[:, 0], m, float(gx - m - 1))
    wp_n[:, 1] = np.clip(wp_n[:, 1], m, float(gy - m - 1))

    return wp_n.astype(np.float32), seg_types
