"""Dataset-sample writer.

Each sample lives in its own directory with a self-contained set of artefacts
so downstream code only needs to read ``metadata.json``::

    run_0000/
        voxels.npz           — compressed boolean voxel grid + heightmap
        trajectory.npy       — (N, 2) float32 waypoints in voxel units
        bev_static.png       — single top-down PNG (start pose, frustums shown)
        bev_video.mp4        — full drive from the BEV, car animated
        cam_<i>_<name>.mp4   — first-person video per *enabled* camera
        metadata.json        — all configs, seed, segment types, files map

A top-level ``manifest.json`` is emitted alongside a multi-run directory.
"""

from __future__ import annotations
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image

from .config import WorldConfig, TrajectoryConfig, RenderConfig, default_cameras
from .trajectory import build_trajectory
from .world import build_world
from .camera import compute_camera_world_pose, render_camera_view
from .bev import compute_heading, render_bev, build_camera_overlays


def _safe_name(s):
    """Filesystem-safe component."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))


# ---------------------------------------------------------------------------
# Single sample
# ---------------------------------------------------------------------------


def generate_sample(
    out_dir, world_cfg, traj_cfg, cameras, render_cfg, run_id=None, progress_fn=None
):
    """Generate one dataset sample.

    Parameters
    ----------
    out_dir     : path-like — sample directory (created if missing).
    world_cfg   : WorldConfig.
    traj_cfg    : TrajectoryConfig.
    cameras     : list[CameraConfig] — all four; only enabled ones produce video.
    render_cfg  : RenderConfig.
    run_id      : optional string; defaults to ``out_dir.name``.
    progress_fn : optional callback ``(frame_idx, n_frames)``.

    Returns the metadata dict that was also written to ``metadata.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = out_dir.name

    # --- 1. build trajectory then world (trajectory shapes the road) ------
    trajectory, seg_types = build_trajectory(
        world_cfg.grid_x,
        world_cfg.grid_y,
        seed=traj_cfg.seed,
        n_segments=traj_cfg.n_segments,
        noise_amplitude=traj_cfg.noise_amplitude,
        smoothing_window=traj_cfg.smoothing_window,
        margin=traj_cfg.margin,
    )
    voxels, heights = build_world(
        world_cfg.grid_x,
        world_cfg.grid_y,
        world_cfg.grid_z,
        seed=world_cfg.seed,
        noise_scale=world_cfg.noise_scale,
        max_obstacle_height=world_cfg.max_obstacle_height,
        road_width=world_cfg.road_width,
        trajectory=trajectory,
        shoulder_extra=world_cfg.shoulder_extra,
    )

    # --- 2. voxel grid + trajectory on disk -------------------------------
    np.savez_compressed(out_dir / "voxels.npz", voxels=voxels, heights=heights)
    np.save(out_dir / "trajectory.npy", trajectory)

    # --- 3. static BEV at start pose --------------------------------------
    idx0 = 0
    h0 = compute_heading(trajectory, idx0)
    overlays_static = build_camera_overlays(cameras, trajectory[idx0], h0)
    bev_static = render_bev(
        voxels,
        trajectory,
        idx0,
        h0,
        camera_overlays=overlays_static,
        display_size=render_cfg.bev_display_size,
    )
    Image.fromarray(bev_static).save(out_dir / "bev_static.png")

    # --- 4. per-frame BEV and per-camera images ---------------------------
    n_frames = int(render_cfg.num_frames)
    frame_idxs = np.linspace(0, len(trajectory) - 1, n_frames).astype(int)

    VX, VY, VZ = voxels.shape
    t_far = max(25.0, 0.85 * max(VX, VY))

    enabled_cams = [c for c in cameras if c.enabled]
    bev_frames = []
    cam_frames = {c.idx: [] for c in enabled_cams}

    for fi, idx in enumerate(frame_idxs):
        if progress_fn is not None:
            progress_fn(fi, n_frames)
        wp = trajectory[idx]
        h = compute_heading(trajectory, idx)

        overlays = build_camera_overlays(cameras, wp, h)
        bev_frames.append(
            render_bev(
                voxels,
                trajectory,
                idx,
                h,
                camera_overlays=overlays,
                display_size=render_cfg.bev_display_size,
            )
        )

        for cam in enabled_cams:
            pos, _, R = compute_camera_world_pose(wp, h, cam)
            cam_frames[cam.idx].append(
                render_camera_view(
                    voxels,
                    pos,
                    R,
                    W=render_cfg.img_w,
                    H=render_cfg.img_h,
                    fov_h_deg=cam.fov,
                    t_near=0.2,
                    t_far=t_far,
                    n_samples=render_cfg.n_samples,
                )
            )

    # --- 5. write video files ---------------------------------------------
    bev_video_name = "bev_video.mp4"
    imageio.mimsave(
        out_dir / bev_video_name,
        bev_frames,
        fps=int(render_cfg.fps),
        macro_block_size=1,
    )

    cam_video_files: dict = {}
    for cam in enabled_cams:
        fname = f"cam_{cam.idx}_{_safe_name(cam.name)}.mp4"
        imageio.mimsave(
            out_dir / fname,
            cam_frames[cam.idx],
            fps=int(render_cfg.fps),
            macro_block_size=1,
        )
        cam_video_files[f"{cam.idx}_{cam.name}"] = fname

    # --- 6. metadata ------------------------------------------------------
    diffs = np.diff(trajectory, axis=0)
    approx_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    meta = {
        "run_id": str(run_id),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "world": asdict(world_cfg),
        "trajectory_params": asdict(traj_cfg),
        "trajectory_info": {
            "num_waypoints": int(trajectory.shape[0]),
            "segment_types": list(seg_types),
            "length_voxels_approx": approx_length,
            "x_range": [float(trajectory[:, 0].min()), float(trajectory[:, 0].max())],
            "y_range": [float(trajectory[:, 1].min()), float(trajectory[:, 1].max())],
            "frame_indices": [int(i) for i in frame_idxs],
        },
        "cameras": [c.to_dict() for c in cameras],
        "render": asdict(render_cfg),
        "voxel_shape": [int(x) for x in voxels.shape],
        "files": {
            "voxels": "voxels.npz",
            "trajectory": "trajectory.npy",
            "bev_static": "bev_static.png",
            "bev_video": bev_video_name,
            "camera_videos": cam_video_files,
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# ---------------------------------------------------------------------------
# Multi-sample driver
# ---------------------------------------------------------------------------


def generate_dataset(
    out_root,
    n_samples,
    base_seed=100,
    world_cfg=None,
    traj_cfg=None,
    cameras=None,
    render_cfg=None,
    vary_seeds=True,
    progress_fn=None,
    n_jobs=-1,
    verbose=10,
    backend="loky",
):
    """Generate N samples into ``out_root`` as ``run_0000/``, ``run_0001/``, ...

    With ``vary_seeds=True`` (default) each run derives new world- and
    trajectory-seeds from ``base_seed + i`` / ``base_seed + i + 1000``. The
    non-seed fields of the passed configs are preserved.

    Parameters
    ----------
    n_jobs
        Number of parallel worker processes. ``-1`` uses all cores; ``1``
        runs serially (and is the only mode where ``progress_fn`` is honoured,
        since the nested per-frame callback can't cross process boundaries).
    verbose
        Verbosity passed to :class:`joblib.Parallel`. ``0`` silent,
        ``>=1`` reports per-task completion.
    backend
        joblib backend — ``"loky"`` (default, process-based) is the right
        choice for this CPU-heavy workload; ``"threading"`` is available for
        debugging but will not actually parallelise the Python orchestration.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    world_cfg = world_cfg or WorldConfig()
    traj_cfg = traj_cfg or TrajectoryConfig()
    cameras = cameras or default_cameras()
    render_cfg = render_cfg or RenderConfig()

    # Resolve every per-run config up front so workers receive fully-formed
    # task tuples and seeds remain deterministic regardless of execution order.
    tasks = []
    for i in range(int(n_samples)):
        run_dir = out_root / f"run_{i:04d}"
        if vary_seeds:
            wc = WorldConfig(**{**asdict(world_cfg), "seed": int(base_seed) + i})
            tc = TrajectoryConfig(
                **{**asdict(traj_cfg), "seed": int(base_seed) + i + 1000}
            )
        else:
            wc, tc = world_cfg, traj_cfg
        tasks.append((run_dir, wc, tc, f"run_{i:04d}"))

    if n_jobs == 1:
        # Serial path — preserves the fine-grained per-frame progress_fn.
        run_metas = []
        for i, (run_dir, wc, tc, run_id) in enumerate(tasks):

            def _frame_cb(fi, nf, _i=i):
                if progress_fn is not None:
                    progress_fn(_i, int(n_samples), fi, nf)

            run_metas.append(
                generate_sample(
                    run_dir,
                    wc,
                    tc,
                    cameras,
                    render_cfg,
                    run_id=run_id,
                    progress_fn=_frame_cb if progress_fn is not None else None,
                )
            )
    else:
        # Parallel path — joblib's verbose handles progress; the fine-grained
        # progress_fn is skipped since it can't cross process boundaries.
        from joblib import Parallel, delayed

        run_metas = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(generate_sample)(
                run_dir,
                wc,
                tc,
                cameras,
                render_cfg,
                run_id=run_id,
            )
            for (run_dir, wc, tc, run_id) in tasks
        )

    with open(out_root / "manifest.json", "w") as f:
        json.dump(
            {
                "n_samples": int(n_samples),
                "base_seed": int(base_seed),
                "vary_seeds": bool(vary_seeds),
                "n_jobs": int(n_jobs),
                "runs": [m["run_id"] for m in run_metas],
            },
            f,
            indent=2,
        )

    return run_metas
