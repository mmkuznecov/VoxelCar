"""Generate a world-model training dataset with incremental shard checkpoints.

One rendered frame per simulation step. Saves the exact (turn_cmd, speed_cmd)
used to transition from frame t to frame t+1. Action space matches
VoxelCarRLEnv so policies trained in one world carry over cleanly.

Three data policies are mixed:

    expert          - A* planner on GT occupancy (no OccNet in the loop)
    noisy_expert    - expert with Gaussian noise on the turn command
    random_safe     - random action; rejected if it would cause immediate collision

Incremental output layout during collection:

    <out-dir>/
        shards/
            shard_0000/
                images.uint8.npy       # (n_i, 3, H, W)
                actions.float32.npy    # (n_i, 2)
                states.float32.npy     # (n_i, 4)
                shard_meta.json        # run list + episode indices in this shard
            shard_0001/
                ...
            shards_index.json          # list of shards written so far

Shards are written atomically: each shard is staged in <shard>.partial/ and
renamed to <shard>/ once all files are on disk, so a crash never leaves a
half-promoted shard.

After all episodes are collected (or with --merge-only), shards are streamed
into the final flat layout expected by the WM training code:

    <out-dir>/
        index.json
        images.uint8.npy
        actions.float32.npy
        states.float32.npy

Resume: on restart, any episode index already present in some shard's
`episode_indices` is skipped, so re-running the same command picks up where
it left off with no lost work.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from voxel_car import (
    CameraConfig,
    EgoGridConfig,
    SCENARIO_PRESETS,
    check_collision,
    compute_camera_world_pose,
    compute_fov_mask,
    generate_scenario,
    plan_next_step,
    render_camera_view,
    sample_world_voxels_to_ego,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signed_turn(h_from, h_to):
    """Signed rotation angle (rad) from unit vector h_from to h_to."""
    ax, ay = float(h_from[0]), float(h_from[1])
    bx, by = float(h_to[0]), float(h_to[1])
    return math.atan2(ax * by - ay * bx, ax * bx + ay * by)


def _rotate_heading(heading, dtheta):
    hx, hy = float(heading[0]), float(heading[1])
    c, s = math.cos(dtheta), math.sin(dtheta)
    nx = hx * c - hy * s
    ny = hx * s + hy * c
    n = math.hypot(nx, ny) + 1e-12
    return np.array([nx / n, ny / n], dtype=np.float32)


def _speed_frac_to_cmd(frac, min_speed_fraction):
    """Map step fraction in [min_frac, 1] -> speed_cmd in [-1, 1]."""
    frac = float(np.clip(frac, min_speed_fraction, 1.0))
    denom = max(1.0 - float(min_speed_fraction), 1e-6)
    u = (frac - float(min_speed_fraction)) / denom
    return float(np.clip(2.0 * u - 1.0, -1.0, 1.0))


def _cmd_to_step_size(speed_cmd, step_size, min_speed_fraction):
    u = (float(speed_cmd) + 1.0) * 0.5
    frac = float(min_speed_fraction) + u * (1.0 - float(min_speed_fraction))
    return float(step_size) * float(frac)


def _render(voxels, pos, heading, cam_cfg, H, W, t_far, n_samples):
    cam_pos_w, _, cam_R = compute_camera_world_pose(pos, heading, cam_cfg)
    return render_camera_view(
        voxels,
        cam_pos_w,
        cam_R,
        W=W,
        H=H,
        fov_h_deg=cam_cfg.fov,
        t_near=0.2,
        t_far=t_far,
        n_samples=int(n_samples),
    )


# ---------------------------------------------------------------------------
# Policies -- each returns (turn_cmd, speed_cmd) in [-1, 1]
# ---------------------------------------------------------------------------


def _expert_cmd(
    scen,
    pos,
    heading,
    ego_cfg,
    cam_cfg,
    fov_mask,
    step_size,
    max_turn_deg,
    min_speed_fraction,
):
    """A* expert over GT occupancy."""
    VX, VY, _VZ = scen.voxels.shape
    gt_occ = sample_world_voxels_to_ego(scen.voxels, pos, heading, ego_cfg)
    result = plan_next_step(
        gt_occ,
        ego_cfg,
        tuple(pos),
        tuple(heading),
        scen.exit_xy,
        fov_mask=fov_mask,
        step_size=step_size,
        max_turn_deg=max_turn_deg,
        lookahead_cells=3,
        inflate=2,
        close_range_cells=3,
        soft_cost_weight=4.0,
        heading_penalty=0.3,
        forward_bias=0.7,
        world_shape=(VX, VY),
        world_margin=2.0,
    )
    max_turn = math.radians(float(max_turn_deg))
    dtheta = _signed_turn(heading, result.target_heading_xy)
    turn_cmd = float(np.clip(dtheta / max_turn, -1.0, 1.0))
    frac = float(result.step_size) / max(step_size, 1e-6)
    speed_cmd = _speed_frac_to_cmd(frac, min_speed_fraction)
    return turn_cmd, speed_cmd


def _noisy_expert_cmd(rng, noise_turn_std_deg, *args, **kwargs):
    turn_cmd, speed_cmd = _expert_cmd(*args, **kwargs)
    max_turn_deg = kwargs.get("max_turn_deg")
    noise = rng.normal(0.0, float(noise_turn_std_deg) / float(max_turn_deg))
    turn_cmd = float(np.clip(turn_cmd + noise, -1.0, 1.0))
    return turn_cmd, speed_cmd


def _random_safe_cmd(
    rng,
    scen,
    pos,
    heading,
    step_size,
    max_turn_deg,
    min_speed_fraction,
):
    """Random action, resampled up to 3 times if it would collide immediately."""
    for _ in range(3):
        turn_cmd = float(rng.uniform(-1.0, 1.0))
        speed_cmd = float(rng.uniform(-1.0, 1.0))
        dtheta = turn_cmd * math.radians(float(max_turn_deg))
        new_h = _rotate_heading(heading, dtheta)
        eff = _cmd_to_step_size(speed_cmd, step_size, min_speed_fraction)
        new_pos = np.array(pos, dtype=np.float32) + eff * new_h
        VX, VY, _ = scen.voxels.shape
        if not (0.0 <= new_pos[0] < VX and 0.0 <= new_pos[1] < VY):
            continue
        if check_collision(scen.voxels, new_pos, new_h):
            continue
        return turn_cmd, speed_cmd
    return 0.0, -0.8


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------


def _collect_episode(
    seed,
    preset,
    policy_name,
    rng,
    ego_cfg,
    cam_cfg,
    image_hw,
    max_steps,
    step_size,
    max_turn_deg,
    min_speed_fraction,
    noise_turn_std_deg,
    n_ray_samples,
):
    """Returns (images(N,3,H,W)uint8, actions(N,2)f32, states(N,4)f32) or None."""
    scen = generate_scenario(seed=int(seed), preset=str(preset))
    pos = np.array(scen.entry_xy, dtype=np.float32)
    heading = np.array(scen.entry_heading, dtype=np.float32)
    heading /= np.linalg.norm(heading) + 1e-12

    VX, VY, _VZ = scen.voxels.shape
    H, W = image_hw
    t_far = max(25.0, 0.85 * max(VX, VY))

    fov_mask = compute_fov_mask(
        ego_cfg,
        cam_cfg,
        image_w=W,
        image_h=H,
        margin_deg=3.0,
    )

    imgs, acts, states = [], [], []
    stuck_window = 15
    pos_history = []

    for t in range(int(max_steps)):
        # Render BEFORE acting so the saved image is the state at step t.
        img = _render(scen.voxels, pos, heading, cam_cfg, H, W, t_far, n_ray_samples)

        if policy_name == "expert":
            turn_cmd, speed_cmd = _expert_cmd(
                scen,
                pos,
                heading,
                ego_cfg,
                cam_cfg,
                fov_mask,
                step_size=step_size,
                max_turn_deg=max_turn_deg,
                min_speed_fraction=min_speed_fraction,
            )
        elif policy_name == "noisy_expert":
            turn_cmd, speed_cmd = _noisy_expert_cmd(
                rng,
                noise_turn_std_deg,
                scen,
                pos,
                heading,
                ego_cfg,
                cam_cfg,
                fov_mask,
                step_size=step_size,
                max_turn_deg=max_turn_deg,
                min_speed_fraction=min_speed_fraction,
            )
        else:
            turn_cmd, speed_cmd = _random_safe_cmd(
                rng,
                scen,
                pos,
                heading,
                step_size,
                max_turn_deg,
                min_speed_fraction,
            )

        # Integrate with EXACTLY the command we'll save.
        dtheta = float(turn_cmd) * math.radians(float(max_turn_deg))
        new_heading = _rotate_heading(heading, dtheta)
        eff_step = _cmd_to_step_size(speed_cmd, step_size, min_speed_fraction)
        new_pos = np.array(pos, dtype=np.float32) + eff_step * new_heading

        oob = not (0.0 <= new_pos[0] < VX and 0.0 <= new_pos[1] < VY)
        col = (not oob) and check_collision(scen.voxels, new_pos, new_heading)
        goal_dist = math.hypot(
            new_pos[0] - scen.exit_xy[0], new_pos[1] - scen.exit_xy[1]
        )

        imgs.append(img.transpose(2, 0, 1))  # (3, H, W) uint8
        acts.append(np.array([turn_cmd, speed_cmd], dtype=np.float32))
        states.append(
            np.array([pos[0], pos[1], heading[0], heading[1]], dtype=np.float32)
        )

        pos_history.append(tuple(new_pos))
        if oob or col:
            break
        pos, heading = new_pos, new_heading
        if goal_dist < 3.0:
            break

        if len(pos_history) > stuck_window:
            recent = pos_history[-stuck_window:]
            travelled = sum(
                math.hypot(
                    recent[i][0] - recent[i - 1][0], recent[i][1] - recent[i - 1][1]
                )
                for i in range(1, len(recent))
            )
            if travelled < 3.0:
                break

    if len(imgs) < 8:
        return None

    return (
        np.stack(imgs).astype(np.uint8),
        np.stack(acts).astype(np.float32),
        np.stack(states).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------


def _shards_root(out_root: Path) -> Path:
    return out_root / "shards"


def _shards_index_path(out_root: Path) -> Path:
    return _shards_root(out_root) / "shards_index.json"


def _load_shards_index(out_root: Path) -> list:
    p = _shards_index_path(out_root)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("shards", [])


def _save_shards_index(out_root: Path, shards: list) -> None:
    """Write shards_index.json atomically (.tmp in same dir, then rename)."""
    _shards_root(out_root).mkdir(parents=True, exist_ok=True)
    p = _shards_index_path(out_root)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"shards": shards}, indent=2))
    tmp.replace(p)


def _discover_done_episodes(out_root: Path) -> set:
    """Recover the set of episode indices already captured in existing shards."""
    done = set()
    for s in _load_shards_index(out_root):
        shard_dir = out_root / "shards" / s
        meta_path = shard_dir / "shard_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        for ep in meta.get("episode_indices", []):
            done.add(int(ep))
    return done


def _write_shard(out_root: Path, shard_id: int, buffer: dict) -> str:
    """Write one shard atomically by staging in <shard>.partial/ and renaming.

    Why a partial directory and not per-file .tmp suffixes:
        np.save("foo.npy.tmp", arr) silently writes to "foo.npy.tmp.npy"
        because numpy auto-appends ".npy" when the path doesn't already end
        in ".npy". Renaming the whole directory once everything is on disk
        sidesteps the problem and gives directory-level atomicity for free.
    """
    name = f"shard_{shard_id:04d}"
    final_dir = _shards_root(out_root) / name
    staging_dir = _shards_root(out_root) / f"{name}.partial"

    # Clean up any leftover staging directory from a prior aborted attempt.
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)

    imgs_flat = np.concatenate(buffer["images"], axis=0).astype(np.uint8)
    acts_flat = np.concatenate(buffer["actions"], axis=0).astype(np.float32)
    states_flat = np.concatenate(buffer["states"], axis=0).astype(np.float32)

    # Final filenames -- np.save will NOT append a second ".npy" because
    # these paths already end in ".npy".
    np.save(staging_dir / "images.uint8.npy", imgs_flat)
    np.save(staging_dir / "actions.float32.npy", acts_flat)
    np.save(staging_dir / "states.float32.npy", states_flat)

    meta = {
        "shard_id": int(shard_id),
        "n_frames": int(imgs_flat.shape[0]),
        "image_shape": list(imgs_flat.shape[1:]),
        "n_episodes": int(len(buffer["runs"])),
        "runs": buffer["runs"],
        "episode_indices": list(map(int, buffer["episode_indices"])),
    }
    (staging_dir / "shard_meta.json").write_text(json.dumps(meta, indent=2))

    # If, for any reason, a final directory with this name already exists,
    # treat it as a leftover and remove it before promotion.
    if final_dir.exists():
        shutil.rmtree(final_dir)
    staging_dir.rename(final_dir)

    return name


def _flush_if_ready(
    out_root: Path,
    buffer: dict,
    shards: list,
    shard_id: int,
    force: bool,
    shard_every: int,
) -> int:
    """If the buffer has enough runs (or force=True), write a shard and reset.

    Returns the next shard id.
    """
    if not buffer["runs"]:
        return shard_id
    if (not force) and len(buffer["runs"]) < int(shard_every):
        return shard_id

    name = _write_shard(out_root, shard_id, buffer)
    shards.append(name)
    _save_shards_index(out_root, shards)
    n_frames = sum(r["n_frames"] for r in buffer["runs"])
    print(f"  -> wrote {name}  ({len(buffer['runs'])} episodes, " f"{n_frames} frames)")

    buffer["images"].clear()
    buffer["actions"].clear()
    buffer["states"].clear()
    buffer["runs"].clear()
    buffer["episode_indices"].clear()
    return shard_id + 1


# ---------------------------------------------------------------------------
# Merge shards -> final flat layout
# ---------------------------------------------------------------------------


def _merge_shards(
    out_root: Path,
    cleanup_shards: bool,
    ego_cfg: EgoGridConfig,
    cam_cfg: CameraConfig,
    extra_args: dict,
) -> None:
    shards = _load_shards_index(out_root)
    if not shards:
        print("No shards to merge.")
        return

    shard_dirs = [_shards_root(out_root) / s for s in shards]

    # Pass 1: compute total size and rewrite per-run start_frame offsets.
    total_frames = 0
    runs_meta: list = []
    image_shape = None
    for sd in shard_dirs:
        meta = json.loads((sd / "shard_meta.json").read_text())
        if image_shape is None:
            image_shape = list(meta["image_shape"])
        elif image_shape != meta["image_shape"]:
            raise SystemExit(
                f"image shape mismatch between shards: "
                f"{image_shape} vs {meta['image_shape']} at {sd}"
            )
        for r in meta["runs"]:
            r2 = dict(r)
            r2["start_frame"] = int(total_frames)
            total_frames += int(r["n_frames"])
            runs_meta.append(r2)

    if total_frames == 0:
        raise SystemExit("shards contain zero frames; nothing to merge")

    C, H, W = image_shape
    print(f"Merging {len(shards)} shards -> {total_frames} frames " f"({C}x{H}x{W})")

    # Pass 2: stream into output memmaps. Use directory-level atomicity again
    # by staging files in a sibling .merge_tmp/ directory.
    stage_dir = out_root / ".merge_tmp"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=False)

    images_out = np.lib.format.open_memmap(
        stage_dir / "images.uint8.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total_frames, C, H, W),
    )
    actions_out = np.lib.format.open_memmap(
        stage_dir / "actions.float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, 2),
    )
    states_out = np.lib.format.open_memmap(
        stage_dir / "states.float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, 4),
    )

    cursor = 0
    for sd in tqdm(shard_dirs, desc="merge"):
        imgs = np.load(sd / "images.uint8.npy", mmap_mode="r")
        acts = np.load(sd / "actions.float32.npy", mmap_mode="r")
        sts = np.load(sd / "states.float32.npy", mmap_mode="r")
        n = int(imgs.shape[0])
        images_out[cursor : cursor + n] = imgs[:]
        actions_out[cursor : cursor + n] = acts[:]
        states_out[cursor : cursor + n] = sts[:]
        cursor += n

    images_out.flush()
    actions_out.flush()
    states_out.flush()
    # Drop memmap handles before renaming (Windows is picky about this).
    del images_out, actions_out, states_out

    index = {
        "n_frames": int(total_frames),
        "image_shape": [C, H, W],
        "action_dim": 2,
        "ego_cfg": ego_cfg.to_dict(),
        "camera": cam_cfg.to_dict(),
        "runs": runs_meta,
        "args": extra_args,
        "n_shards": int(len(shards)),
    }
    (stage_dir / "index.json").write_text(json.dumps(index, indent=2))

    # Promote staged files to the final names.
    for fname in (
        "images.uint8.npy",
        "actions.float32.npy",
        "states.float32.npy",
        "index.json",
    ):
        src = stage_dir / fname
        dst = out_root / fname
        if dst.exists():
            dst.unlink()
        src.rename(dst)
    shutil.rmtree(stage_dir, ignore_errors=True)
    print(f"Wrote {out_root / 'index.json'}")

    if cleanup_shards:
        shutil.rmtree(_shards_root(out_root), ignore_errors=True)
        print(f"Removed shard directory {_shards_root(out_root)}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", default="./wm_data")
    p.add_argument("--n-episodes", type=int, default=2000)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--img-w", type=int, default=128)
    p.add_argument("--img-h", type=int, default=128)
    p.add_argument("--n-ray-samples", type=int, default=150)

    p.add_argument("--preset-mix", default="easy,winding,narrow,tall_obstacles")
    p.add_argument(
        "--policy-mix", default="expert:0.5,noisy_expert:0.3,random_safe:0.2"
    )
    p.add_argument("--noise-turn-std-deg", type=float, default=3.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--max-turn-deg", type=float, default=15.0)
    p.add_argument("--min-speed-fraction", type=float, default=0.10)

    # Sharding / merge.
    p.add_argument(
        "--shard-every",
        type=int,
        default=500,
        help="Write a shard checkpoint every N successful episodes.",
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip generation, just merge existing shards.",
    )
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip the final merge step (shards only).",
    )
    p.add_argument(
        "--cleanup-shards",
        action="store_true",
        help="After a successful merge, delete the shards/ directory.",
    )

    a = p.parse_args(argv)
    out_root = Path(a.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    presets = [s for s in a.preset_mix.split(",") if s.strip() in SCENARIO_PRESETS]
    if not presets:
        raise SystemExit(f"no valid presets in {a.preset_mix}")
    policy_names, weights = [], []
    for tok in a.policy_mix.split(","):
        name, _, w = tok.partition(":")
        if name.strip() not in ("expert", "noisy_expert", "random_safe"):
            raise SystemExit(f"unknown policy '{name}'")
        policy_names.append(name.strip())
        weights.append(float(w))
    weights = np.array(weights) / sum(weights)

    ego_cfg = EgoGridConfig(d_x=20, d_y=16, d_z=12, resolution=1.0)
    cam_cfg = CameraConfig(
        idx=0,
        name="front",
        enabled=True,
        fwd=1.5,
        rgt=0.0,
        height=2.0,
        yaw=0,
        fov=75,
    )

    if a.merge_only:
        _merge_shards(out_root, bool(a.cleanup_shards), ego_cfg, cam_cfg, vars(a))
        return

    print(f"policies: {list(zip(policy_names, weights.tolist()))}")
    print(f"presets : {presets}")
    print(f"shards  : every {int(a.shard_every)} successful episodes")

    # Resume support.
    shards = _load_shards_index(out_root)
    done = _discover_done_episodes(out_root)
    if done:
        print(
            f"resume : found {len(done)} episodes already captured in "
            f"{len(shards)} existing shards"
        )
    next_shard_id = len(shards)

    rng = np.random.RandomState(int(a.base_seed))

    buffer = {
        "images": [],
        "actions": [],
        "states": [],
        "runs": [],
        "episode_indices": [],
    }
    t0 = time.time()

    try:
        for i in tqdm(range(int(a.n_episodes)), desc="episodes"):
            if i in done:
                continue

            preset = presets[i % len(presets)]
            policy = rng.choice(policy_names, p=weights)
            seed = int(a.base_seed) + i

            try:
                out = _collect_episode(
                    seed=seed,
                    preset=preset,
                    policy_name=policy,
                    rng=rng,
                    ego_cfg=ego_cfg,
                    cam_cfg=cam_cfg,
                    image_hw=(int(a.img_h), int(a.img_w)),
                    max_steps=int(a.max_steps),
                    step_size=float(a.step_size),
                    max_turn_deg=float(a.max_turn_deg),
                    min_speed_fraction=float(a.min_speed_fraction),
                    noise_turn_std_deg=float(a.noise_turn_std_deg),
                    n_ray_samples=int(a.n_ray_samples),
                )
            except Exception as exc:
                print(f"  ! episode {i} (seed {seed}) failed: {exc}")
                out = None

            if out is not None:
                imgs, acts, sts = out
                N = int(imgs.shape[0])
                buffer["images"].append(imgs)
                buffer["actions"].append(acts)
                buffer["states"].append(sts)
                # start_frame is recomputed at merge time; sentinel here.
                buffer["runs"].append(
                    {
                        "run_id": f"run_{i:05d}",
                        "seed": int(seed),
                        "preset": preset,
                        "policy": str(policy),
                        "start_frame": -1,
                        "n_frames": int(N),
                    }
                )
                buffer["episode_indices"].append(int(i))

            next_shard_id = _flush_if_ready(
                out_root,
                buffer,
                shards,
                next_shard_id,
                force=False,
                shard_every=int(a.shard_every),
            )

        # Final flush for any remainder.
        next_shard_id = _flush_if_ready(
            out_root,
            buffer,
            shards,
            next_shard_id,
            force=True,
            shard_every=int(a.shard_every),
        )

    except KeyboardInterrupt:
        print("\n! interrupted -- flushing buffered episodes to a shard ...")
        _flush_if_ready(
            out_root,
            buffer,
            shards,
            next_shard_id,
            force=True,
            shard_every=int(a.shard_every),
        )
        print("   done. you can rerun the same command to resume.")
        raise

    n_frames_total = 0
    for s in shards:
        meta_path = out_root / "shards" / s / "shard_meta.json"
        if meta_path.exists():
            n_frames_total += int(json.loads(meta_path.read_text())["n_frames"])
    print(
        f"\nFinished: {len(shards)} shards, {n_frames_total} frames total, "
        f"{time.time() - t0:.1f}s"
    )

    if a.no_merge:
        print("skipping merge (--no-merge). Run with --merge-only later.")
        return

    _merge_shards(out_root, bool(a.cleanup_shards), ego_cfg, cam_cfg, vars(a))


if __name__ == "__main__":
    main()
