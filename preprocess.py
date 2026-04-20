"""One-time preprocessing: dataset runs → flat memmapped arrays for training.

Why preprocess
--------------
Random-access training needs O(1) lookup per sample. mp4 frame seeking is
slow and loading + re-extracting per epoch is wasteful. Here we decode every
forward-camera frame once, extract ego-frame GT once, and lay it all out in
contiguous memmapped files::

    <out-dir>/
        images.uint8.npy          (N, 3, H, W)   raw RGB frames
        gt.uint8.npy              (N, Dx, Dy, Dz) 0/1 occupancy under the FOV mask
        mask.uint8.npy            (Dx, Dy, Dz)    shared FOV mask
        index.json                sample metadata + list of (run_id, frame_idx)

With uint8 throughout, sample size is:
    image  = 3·H·W            ≈ 73 KB (for 180×136)
    GT     = Dx·Dy·Dz         ≈ 3.8 KB (for 20×16×12)

For 600 K samples that's ≈ 44 GB of image data, which memmaps cleanly and
streams at full SSD bandwidth during training.

Usage
-----
    python preprocess.py --data-dir ./data/dataset --out-dir ./preprocessed

    # Optional overrides
    python preprocess.py --data-dir ./data/dataset --out-dir ./pp \\
        --ego-dx 20 --ego-dy 16 --ego-dz 12 --margin-deg 3 \\
        --cam-idx 0 --max-runs 100
"""

from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm

from voxel_car import (
    CameraConfig,
    EgoGridConfig,
    compute_fov_mask,
    sample_world_voxels_to_ego,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_heading(trajectory, idx):
    """Unit heading at waypoint idx via central differences (matches src.bev)."""
    n = len(trajectory)
    if idx <= 0:
        d = trajectory[1] - trajectory[0]
    elif idx >= n - 1:
        d = trajectory[-1] - trajectory[-2]
    else:
        d = trajectory[idx + 1] - trajectory[idx - 1]
    return d / (np.linalg.norm(d) + 1e-12)


def _camera_from_metadata_dict(meta_cam):
    """Reconstruct a CameraConfig-like object from metadata.json entry."""
    return CameraConfig(
        idx=int(meta_cam["idx"]),
        name=str(meta_cam["name"]),
        enabled=bool(meta_cam["enabled"]),
        fwd=float(meta_cam["fwd"]),
        rgt=float(meta_cam["rgt"]),
        height=float(meta_cam["height"]),
        yaw=float(meta_cam["yaw"]),
        fov=float(meta_cam["fov"]),
    )


def _load_run(run_dir, cam_idx):
    """Load everything needed from one run. Returns a dict or None on failure."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text())

    cam_entry = None
    for c in meta["cameras"]:
        if int(c["idx"]) == int(cam_idx) and bool(c["enabled"]):
            cam_entry = c
            break
    if cam_entry is None:
        return None  # camera not enabled in this run

    video_name = meta["files"]["camera_videos"].get(f"{cam_idx}_{cam_entry['name']}")
    if video_name is None:
        return None
    video_path = run_dir / video_name
    if not video_path.exists():
        return None

    voxels = np.load(run_dir / "voxels.npz")["voxels"]
    trajectory = np.load(run_dir / "trajectory.npy")
    frame_indices = np.array(meta["trajectory_info"]["frame_indices"])
    return {
        "meta": meta,
        "cam": cam_entry,
        "video": video_path,
        "voxels": voxels,
        "trajectory": trajectory,
        "frame_indices": frame_indices,
        "img_w": int(meta["render"]["img_w"]),
        "img_h": int(meta["render"]["img_h"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        default="./data/dataset",
        help="Root of generated dataset (contains run_XXXX/...).",
    )
    p.add_argument(
        "--out-dir", default="./preprocessed", help="Where to write memmapped arrays."
    )
    p.add_argument(
        "--cam-idx",
        type=int,
        default=0,
        help="Which camera index to use (default 0 = front).",
    )
    p.add_argument("--ego-dx", type=int, default=20)
    p.add_argument("--ego-dy", type=int, default=16)
    p.add_argument("--ego-dz", type=int, default=12)
    p.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Ego voxel resolution in metres (match world grid: 1.0).",
    )
    p.add_argument(
        "--margin-deg",
        type=float,
        default=3.0,
        help="Angular margin added to camera FOV for the mask.",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Debug: cap number of runs processed.",
    )
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    data_dir = Path(a.data_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Discover runs ----
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        run_ids = list(manifest["runs"])
    else:
        run_ids = sorted(
            p.name
            for p in data_dir.iterdir()
            if p.is_dir() and p.name.startswith("run_")
        )

    if a.max_runs is not None:
        run_ids = run_ids[: int(a.max_runs)]
    print(f"Found {len(run_ids)} runs under {data_dir}")

    # ---- First pass: probe one run to determine shapes ----
    probe = None
    for rid in run_ids:
        probe = _load_run(data_dir / rid, a.cam_idx)
        if probe is not None:
            break
    if probe is None:
        raise RuntimeError(
            "No usable run found (check --cam-idx matches an enabled camera)."
        )
    H, W = probe["img_h"], probe["img_w"]
    cam_cfg = _camera_from_metadata_dict(probe["cam"])
    ego_cfg = EgoGridConfig(
        d_x=a.ego_dx, d_y=a.ego_dy, d_z=a.ego_dz, resolution=float(a.resolution)
    )
    print(f"Image: {H}×{W}    Ego grid: {ego_cfg.shape}    Camera: {cam_cfg.name}")

    # ---- Compute the FOV mask (once, shared across all samples) ----
    mask = compute_fov_mask(
        ego_cfg, cam_cfg, image_w=W, image_h=H, margin_deg=float(a.margin_deg)
    )
    print(
        f"FOV mask: {int(mask.sum())}/{mask.size} voxels "
        f"({100*mask.mean():.1f}%) under supervision"
    )

    # ---- Second pass: count total samples (sum num_frames across runs) ----
    valid_run_ids = []
    per_run_info = []  # list of dicts for the loop below
    total_frames = 0
    for rid in tqdm(run_ids, desc="Indexing"):
        r = _load_run(data_dir / rid, a.cam_idx)
        if r is None:
            continue
        nf = int(r["meta"]["render"]["num_frames"])
        if int(r["img_w"]) != W or int(r["img_h"]) != H:
            # Dimension mismatch across runs — skip to keep a uniform memmap.
            continue
        valid_run_ids.append(rid)
        per_run_info.append(
            {
                "run_id": rid,
                "num_frames": nf,
                "frame_offset": total_frames,
            }
        )
        total_frames += nf
    print(f"Valid runs: {len(valid_run_ids)}    total samples: {total_frames}")

    # ---- Allocate memmaps ----
    img_path = out_dir / "images.uint8.npy"
    gt_path = out_dir / "gt.uint8.npy"
    mask_path = out_dir / "mask.uint8.npy"
    images = np.lib.format.open_memmap(
        img_path, mode="w+", dtype=np.uint8, shape=(total_frames, 3, H, W)
    )
    gt = np.lib.format.open_memmap(
        gt_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_frames, ego_cfg.d_x, ego_cfg.d_y, ego_cfg.d_z),
    )
    np.save(mask_path, mask.astype(np.uint8))

    # ---- Main pass: fill memmaps ----
    t0 = time.time()
    sample_index = []  # (run_id, frame_idx_within_run, world_traj_idx)
    for info in tqdm(per_run_info, desc="Processing"):
        rid = info["run_id"]
        r = _load_run(data_dir / rid, a.cam_idx)
        if r is None:
            continue
        base = info["frame_offset"]
        video = imageio.get_reader(str(r["video"]))
        try:
            for fi in range(info["num_frames"]):
                try:
                    frame = video.get_data(fi)  # (H, W, 3) uint8
                except Exception:
                    # Some encoders don't index-seek cleanly; fall back to iter.
                    video.close()
                    video = imageio.get_reader(str(r["video"]))
                    frames_iter = iter(video)
                    for _ in range(fi):
                        next(frames_iter)
                    frame = next(frames_iter)

                # Store as (3, H, W).
                images[base + fi] = frame.transpose(2, 0, 1)

                # GT at the trajectory waypoint that this frame corresponds to.
                traj_idx = int(r["frame_indices"][fi])
                heading = _compute_heading(r["trajectory"], traj_idx)
                egt = sample_world_voxels_to_ego(
                    r["voxels"], r["trajectory"][traj_idx], heading, ego_cfg
                )
                gt[base + fi] = egt.astype(np.uint8)

                sample_index.append(
                    {
                        "run_id": rid,
                        "frame_idx": int(fi),
                        "traj_idx": traj_idx,
                    }
                )
        finally:
            video.close()

    images.flush()
    gt.flush()
    print(f"Wrote {total_frames} samples in {time.time()-t0:.1f}s")

    # ---- Index + config ----
    index = {
        "num_samples": int(total_frames),
        "image_shape": [3, H, W],
        "gt_shape": list(ego_cfg.shape),
        "ego_cfg": ego_cfg.to_dict(),
        "camera": _camera_from_metadata_dict(probe["cam"]).to_dict(),
        "margin_deg": float(a.margin_deg),
        "cam_idx": int(a.cam_idx),
        "files": {
            "images": img_path.name,
            "gt": gt_path.name,
            "mask": mask_path.name,
        },
        "runs": per_run_info,
        "samples": sample_index,
    }
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index written to {out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
