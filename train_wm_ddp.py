"""Distributed (DDP) trainer for LeWM-style JEPA on the voxel_car WM dataset.

Single-GPU usage (works without any DDP setup):

    python train_wm_ddp.py --data-root ./wm_data --epochs 40

Multi-GPU usage with torchrun (preferred):

    torchrun --standalone --nproc_per_node=4 train_wm_ddp.py \\
        --data-root ./wm_data --epochs 40 --batch-size 64

Per-run artifacts written to <out-dir>/<run-name>/:

    args.json                 CLI args
    history.jsonl             per-epoch metrics (one line per epoch)
    samples.png               8 random training frames -- sanity check input
    loss_curve.png            train/val pred + sigreg + total
    rollout_mse.png           open-loop prediction MSE vs step
    eval_report.json          final probe R^2 + prediction MSE
    summary.json              wall time, best val loss, final probe scores
    ckpt_last.pt              after every epoch
    ckpt_best.pt              best val_loss
    tb/                       TensorBoard scalars (if tensorboard installed)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from voxel_car.wm import SIGReg, VoxelCarJEPA, WMDataset

# ---------------------------------------------------------------------------
# DDP setup / utility
# ---------------------------------------------------------------------------


def _is_dist_run() -> bool:
    """True iff torchrun set the relevant env vars."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _setup_dist(backend: str = "nccl"):
    """Initialise the process group from torchrun env vars; return rank info.

    Returns: (rank, world_size, local_rank, is_dist).
    """
    if not _is_dist_run():
        return 0, 1, 0, False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if not dist.is_initialized():
        # NCCL for GPU, gloo as fallback for CPU.
        chosen = backend if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=chosen, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def _cleanup_dist():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return int(rank) == 0


def _all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all ranks, returning a new tensor."""
    if not dist.is_initialized():
        return t
    out = t.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= dist.get_world_size()
    return out


@contextmanager
def _rank0_only(rank: int):
    """Context manager that yields only on rank 0; others get None and skip."""
    yield _is_main(rank)


# ---------------------------------------------------------------------------
# Step / metrics
# ---------------------------------------------------------------------------


def _collate(batch):
    keep = ("images", "actions", "states")
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keep}


def _compute_losses(model, sigreg, batch, lambda_sigreg, device, return_extras=False):
    """One-step forward returning (loss, pred_loss, sig_loss[, extras])."""
    imgs = batch["images"].to(device, non_blocking=True)  # (B, T, 3, H, W)
    acts = batch["actions"].to(device, non_blocking=True)  # (B, T, 2)

    # DDP wraps the model -- the encode/predict_next methods live on the
    # underlying VoxelCarJEPA. With DDP we must still call the wrapped module
    # via model() for the gradient sync to happen for the *parameters used in
    # the forward pass*. We orchestrate it via two .module method calls but
    # keep the gradient-bearing forward inside the DDP wrapper boundary by
    # calling model.module explicitly (DDP on PyTorch >=1.13 hooks all params,
    # not just those reached by a single forward, so this is correct).
    inner = model.module if isinstance(model, DDP) else model

    z = inner.encode(imgs)  # (B, T, D)
    z_hat = inner.predict_next(z[:, :-1], acts[:, :-1])  # (B, T-1, D)
    z_tgt = z[:, 1:]  # (B, T-1, D)

    pred_loss = (z_hat - z_tgt).pow(2).mean()
    sig_loss = sigreg(z.reshape(-1, z.shape[-1]))
    loss = pred_loss + float(lambda_sigreg) * sig_loss

    if not return_extras:
        return loss, pred_loss.detach(), sig_loss.detach()

    with torch.no_grad():
        # Stats useful for diagnosis.
        z_norm = z.norm(dim=-1).mean()
        z_std = z.std(dim=0).mean()  # avg per-feature std across batch
    extras = {"z_norm": z_norm.detach(), "z_std": z_std.detach()}
    return loss, pred_loss.detach(), sig_loss.detach(), extras


# ---------------------------------------------------------------------------
# Open-loop rollout MSE -- the key "is the predictor doing anything" check
# ---------------------------------------------------------------------------


@torch.no_grad()
def _rollout_mse_curve(model, loader, device, max_batches: int = 16):
    """Per-step open-loop prediction MSE on val. Returns (T-1,) numpy array.

    For each subseq of length T, encode all T frames, then *autoregressively*
    predict z_1, z_2, ..., z_{T-1} starting only from z_0 and the action
    sequence -- comparable to inference. Per-step MSE is averaged across
    samples and feature dims.
    """
    inner = model.module if isinstance(model, DDP) else model
    inner.eval()
    accum = None
    cnt = 0
    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break
        imgs = batch["images"].to(device)
        acts = batch["actions"].to(device)
        z = inner.encode(imgs)  # (B, T, D)
        T = z.shape[1]
        if T < 2:
            continue
        # Autoregressive rollout from z_0 with actions a_0..a_{T-2}.
        z0 = z[:, 0]  # (B, D)
        preds = inner.rollout(z0, acts[:, :-1])  # (B, T-1, D)
        targets = z[:, 1:]  # (B, T-1, D)
        per_step = (preds - targets).pow(2).mean(dim=(0, 2))  # (T-1,)
        if accum is None:
            accum = per_step.detach().clone()
        else:
            accum = accum + per_step.detach()
        cnt += 1
    if accum is None:
        return np.zeros((1,), dtype=np.float32)
    return (accum / max(cnt, 1)).cpu().numpy()


# ---------------------------------------------------------------------------
# End-of-training probe (linear regression: latent -> (x,y,hx,hy))
# ---------------------------------------------------------------------------


@torch.no_grad()
def _gather_latents_and_states(model, loader, device, max_batches=None):
    inner = model.module if isinstance(model, DDP) else model
    inner.eval()
    Z, S = [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= int(max_batches):
            break
        imgs = batch["images"].to(device)
        z = inner.encode(imgs).cpu().numpy()
        Z.append(z.reshape(-1, z.shape[-1]))
        S.append(batch["states"].numpy().reshape(-1, 4))
    if not Z:
        return np.zeros((0, 1)), np.zeros((0, 4))
    return np.concatenate(Z), np.concatenate(S)


def _linear_probe_r2(Z_tr, S_tr, Z_te, S_te):
    """Closed-form least squares probe with bias column. Returns dict."""
    Zt1 = np.concatenate([Z_tr, np.ones((Z_tr.shape[0], 1))], axis=1)
    Zv1 = np.concatenate([Z_te, np.ones((Z_te.shape[0], 1))], axis=1)
    W, *_ = np.linalg.lstsq(Zt1, S_tr, rcond=None)
    pred = Zv1 @ W

    ss_res = ((S_te - pred) ** 2).sum(axis=0)
    ss_tot = ((S_te - S_te.mean(0)) ** 2).sum(axis=0) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    return {
        "r2_x": float(r2[0]),
        "r2_y": float(r2[1]),
        "r2_hx": float(r2[2]),
        "r2_hy": float(r2[3]),
        "mse_xy": float(((S_te[:, :2] - pred[:, :2]) ** 2).mean()),
        "n_train": int(len(Z_tr)),
        "n_test": int(len(Z_te)),
    }


# ---------------------------------------------------------------------------
# Plotting (rank 0 only, lazy import so workers don't pull matplotlib)
# ---------------------------------------------------------------------------


def _plot_loss_curves(history, path):
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(epochs, [r["train_pred"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_pred"] for r in history], label="val")
    axes[0].set_title("prediction MSE")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [r["train_sig"] for r in history], label="train")
    axes[1].plot(epochs, [r["val_sig"] for r in history], label="val")
    axes[1].set_title("SIGReg statistic")
    axes[1].set_xlabel("epoch")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()

    axes[2].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[2].plot(epochs, [r["val_loss"] for r in history], label="val")
    axes[2].set_title("total loss")
    axes[2].set_xlabel("epoch")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(str(path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_rollout_curve(curve, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    steps = np.arange(1, len(curve) + 1)
    ax.plot(steps, curve, marker="o")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("prediction MSE")
    ax.set_title("Open-loop autoregressive rollout MSE on val")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_sample_images(dataset, path, n=8, seed=0):
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(int(seed))
    n = min(n, len(dataset))
    idxs = rng.choice(len(dataset), size=n, replace=False)
    fig, axes = plt.subplots(2, n // 2, figsize=(2.2 * (n // 2), 4.5))
    axes = axes.ravel()
    for i, idx in enumerate(idxs):
        s = dataset[int(idx)]
        img = s["images"][0].numpy().transpose(1, 2, 0)
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].axis("off")
        axes[i].set_title(
            f"act={s['actions'][0].numpy().round(2).tolist()}", fontsize=8
        )
    fig.suptitle("Random training samples (frame 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(path), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optional TB writer
# ---------------------------------------------------------------------------


def _make_tb_writer(out_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter(log_dir=str(out_dir / "tb"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser()

    # Data / I/O
    p.add_argument("--data-root", default="./wm_data")
    p.add_argument("--out-dir", default="./wm_runs")
    p.add_argument("--run-name", default=None)

    # Optimisation
    p.add_argument("--T", type=int, default=4)
    p.add_argument(
        "--batch-size", type=int, default=64, help="per-GPU batch size when using DDP."
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-sigreg", type=float, default=0.1)
    p.add_argument(
        "--scale-lr-with-world",
        action="store_true",
        help="multiply --lr by world_size (a common DDP convention).",
    )

    # Model
    p.add_argument("--emb-dim", type=int, default=192)
    p.add_argument("--pred-depth", type=int, default=3)
    p.add_argument("--pred-heads", type=int, default=4)

    # DataLoader
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.05)

    # AMP / DDP
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--sync-bn",
        action="store_true",
        help="convert BatchNorm to SyncBatchNorm under DDP.",
    )

    # Diagnostics
    p.add_argument(
        "--rollout-eval-every",
        type=int,
        default=2,
        help="recompute open-loop rollout curve every N epochs.",
    )
    p.add_argument(
        "--probe-at-end",
        action="store_true",
        help="run a linear probe (x,y,hx,hy) after final epoch.",
    )

    # Resume
    p.add_argument(
        "--resume", default=None, help="path to a previous ckpt_last.pt to resume from."
    )

    a = p.parse_args(argv)

    # ---- distributed init ----
    rank, world_size, local_rank, is_dist = _setup_dist()
    main_proc = _is_main(rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    use_amp = (not a.no_amp) and (device.type == "cuda")

    if main_proc:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(a.out_dir) / (a.run_name or stamp)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(
            json.dumps({**vars(a), "world_size": world_size}, indent=2)
        )
    else:
        out_dir = Path("/tmp/voxel_wm_dummy")  # never used on non-main

    # Make the chosen out_dir visible to all ranks (for ckpt loading on resume).
    if is_dist:
        out_dir_str = [str(out_dir.resolve())] if main_proc else [None]
        dist.broadcast_object_list(out_dir_str, src=0)
        out_dir = Path(out_dir_str[0])

    # Same seeds on all ranks ensure identical splits, init, etc.
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- dataset ----
    full = WMDataset(a.data_root, T=int(a.T), stride=1)
    C, H, W = full.image_shape
    if main_proc:
        print(f"world_size: {world_size}    device: {device}    amp: {use_amp}")
        print(f"dataset: {len(full)} subseqs,  img = {C}x{H}x{W}")

    n_val = max(1, int(round(len(full) * float(a.val_frac))))
    ds_train, ds_val = random_split(
        full,
        [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(0),  # IDENTICAL on all ranks
    )

    if is_dist:
        train_sampler = DistributedSampler(
            ds_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=0,
        )
        val_sampler = DistributedSampler(
            ds_val,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
            seed=0,
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    dl_train = DataLoader(
        ds_train,
        batch_size=int(a.batch_size),
        shuffle=shuffle_train,
        drop_last=True,
        sampler=train_sampler,
        num_workers=int(a.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
        persistent_workers=(int(a.num_workers) > 0),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(a.batch_size),
        shuffle=False,
        drop_last=False,
        sampler=val_sampler,
        num_workers=int(a.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
        persistent_workers=(int(a.num_workers) > 0),
    )

    # Save sample images once (rank 0).
    if main_proc:
        try:
            _plot_sample_images(ds_train.dataset, out_dir / "samples.png", n=8)
        except Exception as e:
            print(f"  (could not write samples.png: {e})")

    # ---- model ----
    model = VoxelCarJEPA(
        image_hw=(H, W),
        emb_dim=int(a.emb_dim),
        pred_depth=int(a.pred_depth),
        pred_heads=int(a.pred_heads),
    ).to(device)

    if is_dist and a.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if is_dist:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    # SIGReg has no learnable parameters; no DDP wrap.
    sigreg = SIGReg(knots=17, num_proj=512).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if main_proc:
        print(f"model: {n_params / 1e6:.2f} M params")

    # ---- optimiser ----
    lr = float(a.lr) * (world_size if a.scale_lr_with_world else 1)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(a.weight_decay),
        betas=(0.9, 0.95),
    )
    sched = CosineAnnealingLR(opt, T_max=int(a.epochs))
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # ---- resume ----
    start_epoch = 1
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        inner = model.module if isinstance(model, DDP) else model
        inner.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            sched.load_state_dict(ck["scheduler"])
        if "scaler" in ck and scaler is not None and ck["scaler"] is not None:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        if main_proc:
            print(f"resumed from {a.resume} at epoch {start_epoch}")

    # ---- training loop ----
    history = []
    best_val = float("inf")
    tb = _make_tb_writer(out_dir) if main_proc else None
    t_start = time.time()

    for epoch in range(start_epoch, int(a.epochs) + 1):

        # Each epoch must shuffle differently across ranks.
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        # ---- train ----
        model.train()
        sums = torch.zeros(5, device=device)  # tot, pred, sig, znorm, zstd
        n_b = 0
        iterator = dl_train
        if main_proc:
            iterator = tqdm(dl_train, desc=f"ep {epoch:3d} train", leave=False)

        for batch in iterator:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, pl, sl, extras = _compute_losses(
                    model,
                    sigreg,
                    batch,
                    a.lambda_sigreg,
                    device,
                    return_extras=True,
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            sums[0] += loss.detach()
            sums[1] += pl
            sums[2] += sl
            sums[3] += extras["z_norm"]
            sums[4] += extras["z_std"]
            n_b += 1

        sched.step()

        # Cross-rank average for printing.
        cnt = torch.tensor([float(n_b)], device=device)
        sums_avg = _all_reduce_mean(sums / cnt)

        train_loss = float(sums_avg[0])
        train_pred = float(sums_avg[1])
        train_sig = float(sums_avg[2])
        train_znorm = float(sums_avg[3])
        train_zstd = float(sums_avg[4])

        # ---- val ----
        model.eval()
        v_sums = torch.zeros(3, device=device)
        n_vb = 0
        with torch.no_grad():
            for batch in dl_val:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss, pl, sl = _compute_losses(
                        model,
                        sigreg,
                        batch,
                        a.lambda_sigreg,
                        device,
                    )
                v_sums[0] += loss.detach()
                v_sums[1] += pl
                v_sums[2] += sl
                n_vb += 1
        v_cnt = torch.tensor([max(float(n_vb), 1.0)], device=device)
        v_avg = _all_reduce_mean(v_sums / v_cnt)
        val_loss = float(v_avg[0])
        val_pred = float(v_avg[1])
        val_sig = float(v_avg[2])

        row = {
            "epoch": int(epoch),
            "train_loss": train_loss,
            "train_pred": train_pred,
            "train_sig": train_sig,
            "train_znorm": train_znorm,
            "train_zstd": train_zstd,
            "val_loss": val_loss,
            "val_pred": val_pred,
            "val_sig": val_sig,
            "lr": float(opt.param_groups[0]["lr"]),
        }

        # Optional rollout curve.
        if (
            a.rollout_eval_every > 0
            and (epoch % int(a.rollout_eval_every) == 0 or epoch == int(a.epochs))
            and main_proc
        ):
            curve = _rollout_mse_curve(model, dl_val, device, max_batches=8)
            row["rollout_mse"] = curve.tolist()

        history.append(row)

        if main_proc:
            print(
                f"[{epoch:3d}] train {train_loss:.4f} "
                f"(pred {train_pred:.4f}  sig {train_sig:.3f}  "
                f"||z||={train_znorm:.2f})    "
                f"val {val_loss:.4f} "
                f"(pred {val_pred:.4f}  sig {val_sig:.3f})"
            )
            with open(out_dir / "history.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")

            if tb is not None:
                tb.add_scalar("train/loss", train_loss, epoch)
                tb.add_scalar("train/pred_mse", train_pred, epoch)
                tb.add_scalar("train/sigreg", train_sig, epoch)
                tb.add_scalar("train/z_norm", train_znorm, epoch)
                tb.add_scalar("val/loss", val_loss, epoch)
                tb.add_scalar("val/pred_mse", val_pred, epoch)
                tb.add_scalar("val/sigreg", val_sig, epoch)
                tb.add_scalar("lr", row["lr"], epoch)

            try:
                _plot_loss_curves(history, out_dir / "loss_curve.png")
                if "rollout_mse" in row:
                    _plot_rollout_curve(
                        np.asarray(row["rollout_mse"]),
                        out_dir / "rollout_mse.png",
                    )
            except Exception as e:
                print(f"  (plotting failed: {e})")

            # Save checkpoints.
            inner = model.module if isinstance(model, DDP) else model
            ck = {
                "model": inner.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "args": vars(a),
                "epoch": int(epoch),
                "image_hw": (int(H), int(W)),
                "emb_dim": int(a.emb_dim),
                "world_size": int(world_size),
            }
            torch.save(ck, out_dir / "ckpt_last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ck, out_dir / "ckpt_best.pt")

        # Make non-main ranks wait for rank-0's writes before next epoch.
        if is_dist:
            dist.barrier()

    # ---- final probe (rank 0 only) ----
    if a.probe_at_end and main_proc:
        print("\nrunning final linear probe (latents -> x, y, hx, hy) ...")
        # Use single-GPU subsets (no DistributedSampler) so we just iterate val.
        dl_probe_tr = DataLoader(
            Subset(ds_train.dataset, ds_train.indices[: int(0.5 * len(ds_train))]),
            batch_size=int(a.batch_size),
            shuffle=False,
            num_workers=int(a.num_workers),
            collate_fn=_collate,
            pin_memory=(device.type == "cuda"),
        )
        dl_probe_te = DataLoader(
            ds_val.dataset,
            batch_size=int(a.batch_size),
            shuffle=False,
            num_workers=int(a.num_workers),
            collate_fn=_collate,
            pin_memory=(device.type == "cuda"),
        )
        Z_tr, S_tr = _gather_latents_and_states(
            model,
            dl_probe_tr,
            device,
            max_batches=128,
        )
        Z_te, S_te = _gather_latents_and_states(
            model,
            dl_probe_te,
            device,
            max_batches=64,
        )
        if len(Z_tr) and len(Z_te):
            probe = _linear_probe_r2(Z_tr, S_tr, Z_te, S_te)
            with open(out_dir / "eval_report.json", "w") as f:
                json.dump(probe, f, indent=2)
            print(json.dumps(probe, indent=2))

    # ---- summary (rank 0 only) ----
    if main_proc:
        summary = {
            "wall_time_seconds": float(time.time() - t_start),
            "best_val_loss": float(best_val),
            "world_size": int(world_size),
            "epochs": int(a.epochs),
        }
        if a.probe_at_end and (out_dir / "eval_report.json").exists():
            summary["probe"] = json.loads((out_dir / "eval_report.json").read_text())
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\ndone. outputs in {out_dir.resolve()}")
        if tb is not None:
            tb.close()

    _cleanup_dist()


if __name__ == "__main__":
    main()
