"""Train a LeWM-style JEPA on the voxel_car WM dataset."""

from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from voxel_car.wm import SIGReg, VoxelCarJEPA, WMDataset


def _collate(batch):
    # Strip non-tensor fields so default collate works.
    keep = ("images", "actions", "states")
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keep}


def step_fn(model, sigreg, batch, lambda_sigreg, device):
    imgs = batch["images"].to(device, non_blocking=True)  # (B, T, 3, H, W)
    acts = batch["actions"].to(device, non_blocking=True)  # (B, T, 2)

    z = model.encode(imgs)  # (B, T, D)
    z_hat = model.predict_next(z[:, :-1], acts[:, :-1])  # (B, T-1, D)
    z_tgt = z[:, 1:]  # (B, T-1, D)

    pred_loss = (z_hat - z_tgt).pow(2).mean()
    sig_loss = sigreg(z.reshape(-1, z.shape[-1]))
    loss = pred_loss + float(lambda_sigreg) * sig_loss
    return loss, pred_loss.detach(), sig_loss.detach()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./wm_data")
    p.add_argument("--out-dir", default="./wm_runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-sigreg", type=float, default=0.1)
    p.add_argument("--emb-dim", type=int, default=192)
    p.add_argument("--pred-depth", type=int, default=3)
    p.add_argument("--pred-heads", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(a.out_dir) / (a.run_name or stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(a), indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not a.no_amp) and (device.type == "cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    full = WMDataset(a.data_root, T=int(a.T), stride=1)
    C, H, W = full.image_shape
    print(f"dataset: {len(full)} subseqs,  img = {C}x{H}x{W}")

    n_val = max(1, int(round(len(full) * float(a.val_frac))))
    ds_train, ds_val = random_split(
        full,
        [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=int(a.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(a.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(a.batch_size),
        shuffle=False,
        num_workers=int(a.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
    )

    model = VoxelCarJEPA(
        image_hw=(H, W),
        emb_dim=int(a.emb_dim),
        pred_depth=int(a.pred_depth),
        pred_heads=int(a.pred_heads),
    ).to(device)
    sigreg = SIGReg(knots=17, num_proj=512).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f} M params")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
        betas=(0.9, 0.95),
    )
    sched = CosineAnnealingLR(opt, T_max=int(a.epochs))
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    best_val = float("inf")
    for epoch in range(1, int(a.epochs) + 1):
        # -- train --
        model.train()
        tr_tot = tr_pred = tr_sig = 0.0
        n_b = 0
        for batch in tqdm(dl_train, desc=f"ep {epoch:3d} train", leave=False):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, pl, sl = step_fn(model, sigreg, batch, a.lambda_sigreg, device)
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
            tr_tot += float(loss)
            tr_pred += float(pl)
            tr_sig += float(sl)
            n_b += 1
        sched.step()

        # -- val --
        model.eval()
        va_tot = va_pred = va_sig = 0.0
        n_vb = 0
        with torch.no_grad():
            for batch in dl_val:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss, pl, sl = step_fn(
                        model, sigreg, batch, a.lambda_sigreg, device
                    )
                va_tot += float(loss)
                va_pred += float(pl)
                va_sig += float(sl)
                n_vb += 1

        row = {
            "epoch": int(epoch),
            "train_loss": tr_tot / max(n_b, 1),
            "train_pred": tr_pred / max(n_b, 1),
            "train_sig": tr_sig / max(n_b, 1),
            "val_loss": va_tot / max(n_vb, 1),
            "val_pred": va_pred / max(n_vb, 1),
            "val_sig": va_sig / max(n_vb, 1),
            "lr": opt.param_groups[0]["lr"],
        }
        with open(out_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[{epoch:3d}] train {row['train_loss']:.4f} "
            f"(pred {row['train_pred']:.4f}  sig {row['train_sig']:.3f})    "
            f"val {row['val_loss']:.4f} "
            f"(pred {row['val_pred']:.4f}  sig {row['val_sig']:.3f})"
        )

        ckpt = {
            "model": model.state_dict(),
            "args": vars(a),
            "epoch": int(epoch),
            "image_hw": (int(H), int(W)),
            "emb_dim": int(a.emb_dim),
        }
        torch.save(ckpt, out_dir / "ckpt_last.pt")
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(ckpt, out_dir / "ckpt_best.pt")

    print(f"done. outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
