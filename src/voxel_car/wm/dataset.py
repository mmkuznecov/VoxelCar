"""Dataset for WM training from the flat layout produced by generate_wm_data.py."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class WMDataset(Dataset):
    """Yields subsequences of length T. Each item has:
    images  : (T, 3, H, W) float in [0, 1]
    actions : (T, 2)       in [-1, 1]
    states  : (T, 4)       world (x, y, hx, hy)  -- for probing only
    run_id  : str
    start   : int          frame index within the run
    """

    def __init__(self, root, T: int = 4, stride: int = 1):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text())
        self.T = int(T)
        self.stride = int(stride)

        self._imgs = np.load(self.root / "images.uint8.npy", mmap_mode="r")
        self._acts = np.load(self.root / "actions.float32.npy", mmap_mode="r")
        self._sts = np.load(self.root / "states.float32.npy", mmap_mode="r")

        self._starts = []  # list of (run_id, flat_start)
        for r in self.index["runs"]:
            s0 = int(r["start_frame"])
            N = int(r["n_frames"])
            if N < self.T:
                continue
            for k in range(0, N - self.T + 1, self.stride):
                self._starts.append((r["run_id"], s0 + k))

    @property
    def image_shape(self):
        return tuple(self.index["image_shape"])

    def __len__(self):
        return len(self._starts)

    def __getitem__(self, i):
        run_id, start = self._starts[int(i)]
        end = start + self.T
        imgs = np.asarray(self._imgs[start:end]).astype(np.float32) / 255.0
        acts = np.asarray(self._acts[start:end]).astype(np.float32)
        sts = np.asarray(self._sts[start:end]).astype(np.float32)
        return {
            "images": torch.from_numpy(imgs),
            "actions": torch.from_numpy(acts),
            "states": torch.from_numpy(sts),
            "run_id": run_id,
            "start": int(start),
        }
