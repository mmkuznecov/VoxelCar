"""Linear probe: WM latents -> (x, y, hx, hy) world state."""

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from voxel_car.wm import VoxelCarJEPA, WMDataset


def _collate(batch):
    keep = ("images", "actions", "states")
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keep}


@torch.no_grad()
def _encode_all(model, loader, device):
    Z, S = [], []
    for b in loader:
        imgs = b["images"].to(device)
        z = model.encode(imgs).cpu().numpy()
        Z.append(z.reshape(-1, z.shape[-1]))
        S.append(b["states"].numpy().reshape(-1, 4))
    return np.concatenate(Z), np.concatenate(S)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="./wm_data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    a = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.ckpt, map_location=device, weights_only=False)
    ca = ckpt["args"]
    H, W = ckpt["image_hw"]

    model = VoxelCarJEPA(
        image_hw=(H, W),
        emb_dim=int(ca["emb_dim"]),
        pred_depth=int(ca["pred_depth"]),
        pred_heads=int(ca["pred_heads"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = WMDataset(a.data_root, T=int(ca["T"]), stride=4)
    n = len(ds)
    n_tr = int(n * 0.8)
    dl_tr = DataLoader(
        Subset(ds, range(n_tr)),
        batch_size=int(a.batch_size),
        num_workers=int(a.num_workers),
        collate_fn=_collate,
    )
    dl_te = DataLoader(
        Subset(ds, range(n_tr, n)),
        batch_size=int(a.batch_size),
        num_workers=int(a.num_workers),
        collate_fn=_collate,
    )

    Zt, St = _encode_all(model, dl_tr, device)
    Zv, Sv = _encode_all(model, dl_te, device)

    # Least squares with bias.
    Zt1 = np.concatenate([Zt, np.ones((Zt.shape[0], 1))], axis=1)
    Zv1 = np.concatenate([Zv, np.ones((Zv.shape[0], 1))], axis=1)
    W_probe, *_ = np.linalg.lstsq(Zt1, St, rcond=None)
    pred = Zv1 @ W_probe

    ss_res = ((Sv - pred) ** 2).sum(axis=0)
    ss_tot = ((Sv - Sv.mean(0)) ** 2).sum(axis=0) + 1e-9
    r2 = 1.0 - ss_res / ss_tot

    report = {
        "n_train": int(len(Zt)),
        "n_test": int(len(Zv)),
        "r2_x": float(r2[0]),
        "r2_y": float(r2[1]),
        "r2_hx": float(r2[2]),
        "r2_hy": float(r2[3]),
        "mse_xy": float(((Sv[:, :2] - pred[:, :2]) ** 2).mean()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
