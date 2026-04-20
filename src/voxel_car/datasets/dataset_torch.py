"""PyTorch Dataset for the preprocessed occupancy data.

Reads the memmapped image / GT arrays produced by ``preprocess.py``. Opens
the memmaps lazily in each worker so fork semantics behave well.

Splits are computed *per run* (not per frame) to avoid leakage — frames from
the same run share the same world and are highly correlated.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class OccupancyDataset(Dataset):
    """Index-driven Dataset over ``<root>/images.uint8.npy`` and ``gt.uint8.npy``."""

    def __init__(self, root, sample_ids=None):
        """
        Parameters
        ----------
        root : path-like
            Directory containing ``index.json``, ``images.uint8.npy``,
            ``gt.uint8.npy``, ``mask.uint8.npy``.
        sample_ids : array-like of int or None
            Subset of sample indices into the underlying memmaps. If None,
            uses all samples. Use :func:`split_by_run` to derive this.
        """
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text())
        if sample_ids is None:
            sample_ids = np.arange(self.index["num_samples"], dtype=np.int64)
        self.sample_ids = np.asarray(sample_ids, dtype=np.int64)
        # Lazy-open in the worker, not here.
        self._images = None
        self._gt = None
        self._mask = None

    # -- Shared constants ----------------------------------------------------

    @property
    def image_shape(self):
        return tuple(self.index["image_shape"])  # (3, H, W)

    @property
    def gt_shape(self):
        return tuple(self.index["gt_shape"])  # (Dx, Dy, Dz)

    @property
    def ego_cfg(self):
        return dict(self.index["ego_cfg"])

    @property
    def camera(self):
        return dict(self.index["camera"])

    def load_mask(self):
        """Return the shared FOV mask as a bool numpy array (Dx, Dy, Dz)."""
        return np.load(self.root / self.index["files"]["mask"]).astype(bool)

    # -- Lazy memmap handles -------------------------------------------------

    def _ensure_open(self):
        if self._images is None:
            self._images = np.load(
                self.root / self.index["files"]["images"], mmap_mode="r"
            )
            self._gt = np.load(self.root / self.index["files"]["gt"], mmap_mode="r")
            self._mask = np.load(self.root / self.index["files"]["mask"])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, i):
        self._ensure_open()
        idx = int(self.sample_ids[int(i)])
        img = np.asarray(self._images[idx]).astype(np.float32) / 255.0  # (3, H, W)
        gt = np.asarray(self._gt[idx]).astype(np.float32)  # (Dx, Dy, Dz)
        return {
            "image": torch.from_numpy(img),
            "gt": torch.from_numpy(gt),
            "sample_idx": idx,
        }


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def split_by_run(index, fractions=(0.8, 0.1, 0.1), seed=0):
    """Split samples into ``(train, val, test)`` index arrays by ``run_id``.

    Frames from the same run go to the same split — otherwise the val/test
    samples would share a world with training samples, inflating metrics.
    """
    runs = [info["run_id"] for info in index["runs"]]
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(len(runs))
    n = len(runs)
    n_tr = int(round(n * float(fractions[0])))
    n_va = int(round(n * float(fractions[1])))
    train_runs = {runs[i] for i in order[:n_tr]}
    val_runs = {runs[i] for i in order[n_tr : n_tr + n_va]}
    # Remaining → test; accounts for rounding.

    train_ids, val_ids, test_ids = [], [], []
    for gi, s in enumerate(index["samples"]):
        rid = s["run_id"]
        if rid in train_runs:
            train_ids.append(gi)
        elif rid in val_runs:
            val_ids.append(gi)
        else:
            test_ids.append(gi)
    return (
        np.asarray(train_ids, dtype=np.int64),
        np.asarray(val_ids, dtype=np.int64),
        np.asarray(test_ids, dtype=np.int64),
    )
