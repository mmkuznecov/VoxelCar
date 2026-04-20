"""BEV visualisation of ego-frame occupancy predictions.

For each test sample we render a compact figure:

  ┌─────────────────┬──────────────────────┐
  │                 │                      │
  │   input image   │    ego BEV panel     │
  │                 │  (forward = up,      │
  │                 │   right = right)     │
  │                 │                      │
  └─────────────────┴──────────────────────┘

The ego BEV panel colours each (D_x, D_y) cell by aggregating the D_z
column occupancy (OR) and comparing against ground truth under the FOV mask:

    GT=1, Pr=1   →  green   (TP)
    GT=1, Pr=0   →  RED     (FN — missed something real)
    GT=0, Pr=1   →  orange  (FP — hallucinated)
    GT=0, Pr=0   →  grey    (correct empty)
    out-of-mask  →  hatched dark grey (not supervised)
"""

from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Colours (RGB 0-1)
_C_TP = (0.30, 0.80, 0.40)
_C_FN = (0.90, 0.20, 0.20)  # red — the user requested red error markers
_C_FP = (0.95, 0.60, 0.20)
_C_TN = (0.85, 0.85, 0.88)
_C_OOMASK = (0.55, 0.55, 0.58)


# ---------------------------------------------------------------------------
# Ego BEV panel
# ---------------------------------------------------------------------------


def _ego_bev_arrays(gt_3d, pred_3d, mask_3d):
    """Collapse along z and build per-cell category map.

    gt_3d, pred_3d : (Dx, Dy, Dz) bool
    mask_3d        : (Dx, Dy, Dz) bool

    Returns
    -------
    rgba : (Dx, Dy, 4) float — per-cell colour for plotting (row=forward, col=right)
    per_cell_stats : dict
    """
    # BEV = any-z occupied
    gt_b = gt_3d.any(axis=-1)
    pred_b = pred_3d.any(axis=-1)
    mask_b = mask_3d.any(axis=-1)

    rgba = np.zeros((gt_b.shape[0], gt_b.shape[1], 4), dtype=np.float32)

    # Default TN
    rgba[..., :3] = _C_TN
    rgba[..., 3] = 1.0

    tp = mask_b & gt_b & pred_b
    fn = mask_b & gt_b & (~pred_b)
    fp = mask_b & (~gt_b) & pred_b

    rgba[tp, :3] = _C_TP
    rgba[fn, :3] = _C_FN
    rgba[fp, :3] = _C_FP

    # Out of mask → hatched grey
    rgba[~mask_b, :3] = _C_OOMASK

    stats = {
        "tp": int(tp.sum()),
        "fn": int(fn.sum()),
        "fp": int(fp.sum()),
        "n_mask": int(mask_b.sum()),
    }
    return rgba, stats


def render_ego_bev(ax, gt_3d, pred_3d, mask_3d, resolution_m=1.0, title=None):
    """Draw the ego BEV error map onto a matplotlib axis.

    Forward (ego_x) is UP in the panel; right (ego_y) is RIGHT.
    """
    rgba, stats = _ego_bev_arrays(gt_3d, pred_3d, mask_3d)
    Dx, Dy = rgba.shape[:2]

    # imshow wants (H, W, 4) with row 0 at top. We want forward=up, so flip ego_x.
    # Ego array is (Dx, Dy); display row = Dx-1-i so that i=Dx-1 shows at top.
    disp = np.flip(rgba, axis=0)  # (Dx rows, Dy cols, 4)
    extent = [-Dy / 2 * resolution_m, Dy / 2 * resolution_m, 0, Dx * resolution_m]
    ax.imshow(
        disp, extent=extent, origin="lower", aspect="equal", interpolation="nearest"
    )

    # Car glyph at origin (below the grid start).
    ax.add_patch(
        patches.Rectangle(
            (-1.2, -2.4),
            2.4,
            2.0,
            facecolor=(0.9, 0.3, 0.3),
            edgecolor="white",
            linewidth=1.2,
        )
    )
    ax.plot([0, 0], [-0.4, 1.0], color="yellow", linewidth=2)

    ax.set_xlabel("ego_y  (right, m)")
    ax.set_ylabel("ego_x  (forward, m)")
    ax.set_xlim(extent[0] - 1, extent[1] + 1)
    ax.set_ylim(-3, extent[3] + 1)
    ax.grid(alpha=0.2)
    if title is not None:
        ax.set_title(title, fontsize=10)

    return stats


# ---------------------------------------------------------------------------
# Combined "sample report" figure
# ---------------------------------------------------------------------------


def render_sample_figure(
    image_chw, gt_3d, pred_3d, mask_3d, sample_idx=None, iou=None, resolution_m=1.0
):
    """Return a matplotlib Figure with input image + ego BEV error panel."""
    fig, axes = plt.subplots(
        1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [1.0, 1.1]}
    )
    ax_img, ax_bev = axes

    # Input image (convert CHW float → HWC uint8 for display).
    img = image_chw
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    ax_img.imshow(img)
    ax_img.set_title("Input: forward camera")
    ax_img.axis("off")

    # Ego BEV error panel.
    title_bits = []
    if sample_idx is not None:
        title_bits.append(f"sample {int(sample_idx)}")
    if iou is not None:
        title_bits.append(f"BEV IoU = {float(iou):.3f}")
    title = "   ".join(title_bits) if title_bits else None
    stats = render_ego_bev(
        ax_bev, gt_3d, pred_3d, mask_3d, resolution_m=resolution_m, title=title
    )

    # Legend
    legend_patches = [
        patches.Patch(color=_C_TP, label=f"TP (correct) ×{stats['tp']}"),
        patches.Patch(color=_C_FN, label=f"FN (missed) ×{stats['fn']}"),
        patches.Patch(color=_C_FP, label=f"FP (imagined) ×{stats['fp']}"),
        patches.Patch(color=_C_OOMASK, label="outside FOV"),
    ]
    ax_bev.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()
    return fig, stats


def save_sample_figure(path, *args, **kwargs):
    fig, stats = render_sample_figure(*args, **kwargs)
    fig.savefig(str(path), dpi=110, bbox_inches="tight")
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------


def plot_curves(history, path):
    """Plot train/val loss and IoU curves from a list of per-epoch dicts."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    ep = [h["epoch"] for h in history]
    axes[0].plot(
        ep, [h["train_loss"] for h in history], label="train", color="tab:blue"
    )
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val", color="tab:orange")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("masked BCE")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [h["val_iou"] for h in history], color="tab:green", marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Validation IoU (occupied class, masked)")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(str(path), dpi=110, bbox_inches="tight")
    plt.close(fig)
