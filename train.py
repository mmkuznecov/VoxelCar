"""Train the monocular occupancy model.

Runs three phases:

  1. Train for --epochs with masked BCE loss, log per-epoch metrics,
     checkpoint the best-val model.
  2. Evaluate the best model on the held-out test set.
  3. Dump N ego-BEV error visualisations (red markers where predictions missed).

Each training run writes everything to a timestamped directory::

    runs/<timestamp>/
        ckpt_best.pt        # best val-IoU checkpoint
        ckpt_last.pt        # final epoch checkpoint (for resume)
        history.jsonl       # per-epoch metrics
        loss_curve.png      # train/val loss + val IoU
        eval_report.json    # final test metrics
        viz/sample_*.png    # error visualisations
"""

from __future__ import annotations
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from voxel_car import (
    OccupancyDataset,
    split_by_run,
    OccNet,
    save_sample_figure,
    plot_curves,
)

# ---------------------------------------------------------------------------
# Loss & metrics
# ---------------------------------------------------------------------------


def masked_bce_loss(logits, target, mask_b, pos_weight=5.0):
    """Masked BCE-with-logits. Mean over masked voxels (per batch)."""
    # logits, target: (B, Dx, Dy, Dz); mask_b: (Dx, Dy, Dz) bool broadcastable.
    pw = torch.as_tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pw, reduction="none"
    )
    m = mask_b.to(loss.dtype)
    # Broadcast mask over the batch dim.
    loss = loss * m
    denom = m.sum() * logits.shape[0] + 1e-6
    return loss.sum() / denom


@torch.no_grad()
def _accumulate_metrics(logits, target, mask_b, agg):
    """Update running TP/FP/FN/TN counts for IoU / accuracy reporting."""
    pred = torch.sigmoid(logits) > 0.5
    gt = target > 0.5
    m = mask_b.bool()
    # broadcast
    m_b = m.unsqueeze(0).expand_as(pred)
    tp = int(((pred & gt) & m_b).sum().item())
    fp = int(((pred & ~gt) & m_b).sum().item())
    fn = int(((~pred & gt) & m_b).sum().item())
    tn = int(((~pred & ~gt) & m_b).sum().item())
    agg["tp"] += tp
    agg["fp"] += fp
    agg["fn"] += fn
    agg["tn"] += tn


def _finalise_metrics(agg):
    tp, fp, fn, tn = agg["tp"], agg["fp"], agg["fn"], agg["tn"]
    total = tp + fp + fn + tn
    iou = tp / (tp + fp + fn + 1e-9)
    acc = (tp + tn) / (total + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return {
        "iou": iou,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model,
    loader,
    optimiser,
    scaler,
    device,
    mask_b,
    pos_weight,
    grad_clip=1.0,
    log_every=50,
):
    model.train()
    total_loss, n_batches = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        img = batch["image"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
            logits = model(img)
            loss = masked_bce_loss(logits, gt, mask_b, pos_weight=pos_weight)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()

        total_loss += float(loss.item())
        n_batches += 1
        if n_batches % log_every == 0:
            pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, mask_b, pos_weight):
    model.eval()
    total_loss, n_batches = 0.0, 0
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for batch in tqdm(loader, desc="eval", leave=False):
        img = batch["image"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        logits = model(img)
        loss = masked_bce_loss(logits, gt, mask_b, pos_weight=pos_weight)
        total_loss += float(loss.item())
        n_batches += 1
        _accumulate_metrics(logits, gt, mask_b, agg)
    m = _finalise_metrics(agg)
    m["loss"] = total_loss / max(n_batches, 1)
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-root",
        default="./preprocessed",
        help="Preprocessed-dataset root (from preprocess.py).",
    )
    p.add_argument(
        "--out-dir",
        default="./runs",
        help="Parent directory for per-run subdirectories.",
    )
    p.add_argument(
        "--run-name", default=None, help="Name suffix; default is timestamp."
    )

    # Optimisation
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--pos-weight",
        type=float,
        default=4.0,
        help="BCE positive-class weight (mitigates class imbalance).",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed-precision (enabled by default on CUDA).",
    )

    # Data
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)

    # Viz
    p.add_argument(
        "--viz-n",
        type=int,
        default=16,
        help="Number of test samples to save error-visualisations for.",
    )

    # Debug
    p.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Cap batches per epoch (for smoke tests).",
    )

    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- Output dir ----
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = a.run_name or stamp
    out_dir = Path(a.out_dir) / name
    (out_dir / "viz").mkdir(parents=True, exist_ok=True)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not a.no_amp) and (device.type == "cuda")
    print(f"device={device}  amp={use_amp}")

    # ---- Dataset & splits ----
    base_ds = OccupancyDataset(a.data_root)
    index = base_ds.index
    Dx, Dy, Dz = base_ds.gt_shape
    print(
        f"preprocessed: {index['num_samples']} samples  "
        f"image {tuple(base_ds.image_shape)}  gt {base_ds.gt_shape}"
    )

    tr_ids, va_ids, te_ids = split_by_run(
        index,
        fractions=(1 - a.val_frac - a.test_frac, a.val_frac, a.test_frac),
        seed=a.split_seed,
    )
    print(
        f"splits (by run): train={len(tr_ids)}  val={len(va_ids)}  test={len(te_ids)}"
    )

    ds_train = OccupancyDataset(a.data_root, sample_ids=tr_ids)
    ds_val = OccupancyDataset(a.data_root, sample_ids=va_ids)
    ds_test = OccupancyDataset(a.data_root, sample_ids=te_ids)

    dl_kwargs = dict(
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(a.num_workers > 0),
    )
    dl_train = DataLoader(ds_train, shuffle=True, drop_last=True, **dl_kwargs)
    dl_val = DataLoader(ds_val, shuffle=False, **dl_kwargs)
    dl_test = DataLoader(ds_test, shuffle=False, **dl_kwargs)

    # ---- Mask (constant across samples) ----
    mask_np = base_ds.load_mask()
    mask_t = torch.from_numpy(mask_np).to(device)
    print(
        f"FOV mask: {int(mask_np.sum())}/{mask_np.size} voxels "
        f"({100*mask_np.mean():.1f}%) supervised"
    )

    # ---- Model ----
    model = OccNet(d_x=Dx, d_y=Dy, d_z=Dz).to(device)
    print(f"model params: {OccNet.num_params(model)/1e6:.2f} M")

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=a.lr, weight_decay=a.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, a.epochs)
    )
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # ---- Train loop ----
    history = []
    best_iou = -1.0
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        t_ep = time.time()
        # Optional cap for smoke tests
        if a.max_train_batches is not None:
            capped = _CappedDataLoader(dl_train, a.max_train_batches)
        else:
            capped = dl_train
        train_loss = train_one_epoch(
            model,
            capped,
            optimiser,
            scaler,
            device,
            mask_t,
            pos_weight=a.pos_weight,
            grad_clip=a.grad_clip,
        )
        scheduler.step()

        val = evaluate(model, dl_val, device, mask_t, pos_weight=a.pos_weight)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_iou": val["iou"],
            "val_acc": val["accuracy"],
            "val_prec": val["precision"],
            "val_rec": val["recall"],
            "lr": float(optimiser.param_groups[0]["lr"]),
            "sec": time.time() - t_ep,
        }
        history.append(row)
        with open(out_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[ep {epoch:3d}]  train {train_loss:.4f}   "
            f"val loss {val['loss']:.4f}  IoU {val['iou']:.3f}  "
            f"P {val['precision']:.3f}  R {val['recall']:.3f}   "
            f"({row['sec']:.1f}s)"
        )

        # Checkpoints
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimiser.state_dict(),
            "history": history,
            "args": vars(a),
            "ego_cfg": base_ds.ego_cfg,
            "camera": base_ds.camera,
        }
        torch.save(ckpt, out_dir / "ckpt_last.pt")
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save(ckpt, out_dir / "ckpt_best.pt")

        plot_curves(history, out_dir / "loss_curve.png")

    print(f"\ntotal training: {time.time() - t0:.1f}s   best val IoU {best_iou:.3f}")

    # ---- Final test pass ----
    best = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    print("\n== Test set ==")
    test = evaluate(model, dl_test, device, mask_t, pos_weight=a.pos_weight)
    test_report = {k: float(v) if isinstance(v, float) else v for k, v in test.items()}
    test_report["best_epoch"] = int(best["epoch"])
    with open(out_dir / "eval_report.json", "w") as f:
        json.dump(test_report, f, indent=2)
    print(json.dumps(test_report, indent=2))

    # ---- Viz (test samples with BEV error maps) ----
    print(f"\n== Dumping {a.viz_n} error visualisations ==")
    _dump_viz(
        model,
        ds_test,
        mask_np,
        device,
        out_dir / "viz",
        n_samples=a.viz_n,
        resolution_m=base_ds.ego_cfg["resolution"],
    )
    print(f"\nAll outputs in {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CappedDataLoader:
    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = int(max_batches)

    def __iter__(self):
        it = iter(self.loader)
        n = 0
        while n < self.max_batches:
            try:
                yield next(it)
            except StopIteration:
                return
            n += 1

    def __len__(self):
        return min(len(self.loader), self.max_batches)


@torch.no_grad()
def _dump_viz(model, test_ds, mask_np, device, viz_dir, n_samples, resolution_m):
    """Render N error visualisations. Picks samples with varying difficulty."""
    model.eval()
    viz_dir = Path(viz_dir)

    # Score each test sample by foreground complexity (rough proxy for "interesting").
    rng = np.random.RandomState(0)
    n = len(test_ds)
    pick = rng.choice(n, size=min(n_samples, n), replace=False)
    pick.sort()

    for rank, i in enumerate(pick):
        sample = test_ds[int(i)]
        img = sample["image"].unsqueeze(0).to(device)
        gt = sample["gt"].numpy().astype(bool)
        logits = model(img)[0].float().cpu().numpy()
        pred = logits > 0.0  # sigmoid>0.5 ≡ logits>0
        # Per-sample IoU (BEV, any-z)
        gt_b = gt.any(-1)
        pr_b = pred.any(-1)
        m_b = mask_np.any(-1)
        tp = int((gt_b & pr_b & m_b).sum())
        u = int(((gt_b | pr_b) & m_b).sum())
        iou = tp / (u + 1e-9)
        path = (
            viz_dir
            / f"sample_{rank:03d}_idx{int(sample['sample_idx']):06d}_iou{iou:.2f}.png"
        )
        save_sample_figure(
            path,
            sample["image"].numpy(),
            gt,
            pred,
            mask_np,
            sample_idx=int(sample["sample_idx"]),
            iou=iou,
            resolution_m=float(resolution_m),
        )


if __name__ == "__main__":
    main()
