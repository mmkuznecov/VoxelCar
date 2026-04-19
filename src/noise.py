"""Multi-octave value noise, self-contained (no external 'noise' dep)."""

import numpy as np


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def value_noise_2d(shape, scale=18, octaves=4, persistence=0.5, seed=42):
    """Multi-octave value noise normalised to [0, 1]. ``shape`` = (H, W).

    Each octave samples a coarse lattice of uniform(0, 1) values and performs
    a smoothstep-interpolated bilinear upsample. Higher octaves double the
    frequency and halve the amplitude (classical fBm).
    """
    rng = np.random.RandomState(int(seed))
    H, W = shape
    out = np.zeros(shape, dtype=np.float32)
    amp, freq, max_amp = 1.0, 1.0, 0.0

    for _ in range(int(octaves)):
        nH = max(2, int(np.ceil(H / scale * freq)) + 1)
        nW = max(2, int(np.ceil(W / scale * freq)) + 1)
        coarse = rng.rand(nH, nW).astype(np.float32)

        yc = np.linspace(0, nH - 1, H, dtype=np.float32)
        xc = np.linspace(0, nW - 1, W, dtype=np.float32)
        yi = np.floor(yc).astype(np.int32)
        xi = np.floor(xc).astype(np.int32)
        yi = np.clip(yi, 0, nH - 2)
        xi = np.clip(xi, 0, nW - 2)
        ty = _smoothstep(yc - yi)
        tx = _smoothstep(xc - xi)

        c00 = coarse[yi[:, None], xi[None, :]]
        c01 = coarse[yi[:, None], xi[None, :] + 1]
        c10 = coarse[yi[:, None] + 1, xi[None, :]]
        c11 = coarse[yi[:, None] + 1, xi[None, :] + 1]

        top = c00 * (1 - tx)[None, :] + c01 * tx[None, :]
        bot = c10 * (1 - tx)[None, :] + c11 * tx[None, :]
        layer = top * (1 - ty)[:, None] + bot * ty[:, None]

        out += amp * layer
        max_amp += amp
        amp *= persistence
        freq *= 2.0

    out /= max_amp
    return np.clip(out, 0.0, 1.0)
