"""CLI dataset generator.

Example
-------
    python generate.py --out-dir ./data --n 5 --base-seed 100

Each run is written to ``./data/run_0000/``, ``./data/run_0001/``, etc. with
voxels, trajectory, BEV still + video, one video per enabled camera, and a
``metadata.json``. A top-level ``manifest.json`` lists every run.
"""

from __future__ import annotations
import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

from voxel_car import (
    WorldConfig,
    TrajectoryConfig,
    RenderConfig,
    default_cameras,
    generate_dataset,
)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a voxel-landscape car-driving dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out-dir", default="./dataset", help="Root directory for dataset runs."
    )
    p.add_argument("--n", type=int, default=5, help="Number of samples to generate.")
    p.add_argument(
        "--base-seed",
        type=int,
        default=100,
        help="Starting seed; run i uses base_seed + i for the world "
        "and base_seed + i + 1000 for the trajectory.",
    )

    # World
    p.add_argument("--grid-size", type=int, default=80)
    p.add_argument("--max-obst-h", type=int, default=14)
    p.add_argument("--road-w", type=int, default=3)
    p.add_argument("--noise-scale", type=float, default=18.0)
    p.add_argument("--shoulder", type=int, default=2)

    # Trajectory
    p.add_argument("--n-segments", type=int, default=6)
    p.add_argument("--traj-noise", type=float, default=0.4)

    # Render
    p.add_argument("--num-frames", type=int, default=40)
    p.add_argument("--img-w", type=int, default=180)
    p.add_argument("--img-h", type=int, default=136)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument(
        "--n-samples", type=int, default=200, help="Ray-march samples per camera ray."
    )

    # Parallelism
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel worker processes (joblib). "
        "-1 = all cores, 1 = serial (keeps per-frame progress).",
    )
    p.add_argument(
        "--verbose",
        type=int,
        default=10,
        help="joblib verbose level (0 silent, 10 per-task progress).",
    )

    # Camera overrides (comma-separated list of 1-based indices to enable).
    p.add_argument(
        "--enable-cams",
        default=None,
        help="Comma-separated 1-based camera indices to enable, "
        "overriding defaults (e.g. '1,3,4'). "
        "If omitted, the default setup is used.",
    )

    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)

    world_cfg = WorldConfig(
        seed=int(a.base_seed),
        grid_size=int(a.grid_size),
        max_obstacle_height=int(a.max_obst_h),
        road_width=int(a.road_w),
        noise_scale=float(a.noise_scale),
        shoulder_extra=int(a.shoulder),
    )
    traj_cfg = TrajectoryConfig(
        seed=int(a.base_seed) + 1000,
        n_segments=int(a.n_segments),
        noise_amplitude=float(a.traj_noise),
    )
    render_cfg = RenderConfig(
        img_w=int(a.img_w),
        img_h=int(a.img_h),
        num_frames=int(a.num_frames),
        fps=int(a.fps),
        n_samples=int(a.n_samples),
    )

    cameras = default_cameras()
    if a.enable_cams is not None:
        enabled = {int(s) - 1 for s in a.enable_cams.split(",") if s.strip()}
        for c in cameras:
            c.enabled = c.idx in enabled

    t0 = time.time()

    def progress(run_i, n_runs, frame_i, n_frames):
        if frame_i in (0, n_frames - 1) or frame_i % max(1, n_frames // 4) == 0:
            elapsed = time.time() - t0
            sys.stdout.write(
                f"\r[run {run_i + 1}/{n_runs}] frame {frame_i + 1}/{n_frames}"
                f"    elapsed {elapsed:6.1f}s"
            )
            sys.stdout.flush()

    print(f"Generating {a.n} samples into {a.out_dir}")
    print(f"  world : {asdict(world_cfg)}")
    print(f"  traj  : {asdict(traj_cfg)}")
    print(f"  render: {asdict(render_cfg)}")
    print(f"  cams  : {[(c.idx, c.name, c.enabled) for c in cameras]}")
    print(f"  n_jobs: {a.n_jobs}   (joblib verbose={a.verbose})")
    print()

    # Fine-grained per-frame callback only works serially; in parallel mode
    # joblib's own verbose output reports per-task completion.
    progress_fn = progress if a.n_jobs == 1 else None

    metas = generate_dataset(
        a.out_dir,
        a.n,
        base_seed=a.base_seed,
        world_cfg=world_cfg,
        traj_cfg=traj_cfg,
        cameras=cameras,
        render_cfg=render_cfg,
        progress_fn=progress_fn,
        n_jobs=a.n_jobs,
        verbose=a.verbose,
    )
    print()
    print(
        f"Done in {time.time() - t0:.1f}s. Wrote {len(metas)} runs to "
        f"{Path(a.out_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
