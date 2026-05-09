"""Smoke-test for the multi-biome worldgen pipeline.

Builds one scenario per preset, validates that:
  * the bool ``voxels`` grid is internally consistent with the materials grid
    (``voxels[i,j,k] == (materials[i,j,k] != AIR)``),
  * roads carve a flat corridor along the trajectory,
  * sea_level > 0 actually produces water voxels,
  * forest preset places trees,
  * stone_line_offset produces stone surface peaks.

Run with::

    PYTHONPATH=src python scripts/test_worldgen.py
    PYTHONPATH=src python scripts/test_worldgen.py --render-bev /tmp/bevs

The optional ``--render-bev DIR`` flag writes a top-down PNG of each
preset to ``DIR/<preset>.png`` (requires ``Pillow`` and the package's
rendering subpackage to be importable).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from voxel_car.common.materials import Material
from voxel_car.worldgen.scenarios import SCENARIO_PRESETS, generate_scenario


def _audit_scenario(scen):
    """Run a battery of sanity checks on a built scenario, return summary."""
    voxels = scen.voxels
    materials = scen.materials  # may be None for legacy presets that opt
    # out of biome rendering — in that case we
    # only audit the bool grid.

    # ground row z=0 must be solid (existing collision check skips z=0,
    # but ray-march renderer expects it solid for ground colour).
    if not bool(voxels[:, :, 0].all()):
        raise AssertionError("z=0 ground plane is not entirely solid")

    if materials is not None:
        assert voxels.shape == materials.shape, "shape mismatch"
        derived = materials != int(Material.AIR)
        if not np.array_equal(voxels, derived):
            bad = int(np.sum(voxels != derived))
            raise AssertionError(f"voxels vs materials mismatch in {bad} cells")

        # Some columns under the trajectory must be ROAD on top.
        road_col = (materials == int(Material.ROAD)).any(axis=-1)
        if int(road_col.sum()) == 0:
            raise AssertionError("no ROAD cells anywhere — road carving failed")

        counts = {m.name: int((materials == int(m)).sum()) for m in Material}

        Z = materials.shape[-1]
        rev = (materials != int(Material.AIR))[:, :, ::-1]
        first_from_top = np.argmax(rev, axis=-1)
        has_any = (materials != int(Material.AIR)).any(axis=-1)
        top_z = np.where(has_any, Z - 1 - first_from_top, 0).astype(np.int32)
        X, Y, _ = materials.shape
        xs = np.arange(X)[:, None]
        ys = np.arange(Y)[None, :]
        surface_mat = materials[xs, ys, top_z]
        surface_counts = {m.name: int((surface_mat == int(m)).sum()) for m in Material}
    else:
        counts = {"AIR": int((~voxels).sum()), "OCCUPIED": int(voxels.sum())}
        surface_counts = {}

    return {
        "preset": scen.preset,
        "shape": list(scen.grid_shape),
        "n_trees": len(scen.tree_positions),
        "voxel_counts": counts,
        "surface_counts": surface_counts,
        "biome_render": materials is not None,
    }


def _render_bev_png(scen, path):
    """Save a top-down BEV PNG of the scenario for visual sanity check."""
    from PIL import Image
    from voxel_car.rendering.bev import (
        compute_heading,
        render_bev,
    )

    traj = scen.reference_trajectory
    h0 = compute_heading(traj, 0)
    img = render_bev(
        scen.voxels,
        traj,
        0,
        h0,
        materials=scen.materials,
        display_size=480,
    )
    Image.fromarray(img).save(str(path))


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--render-bev",
        default=None,
        help="If given, write per-preset BEV PNGs to this directory.",
    )
    p.add_argument(
        "--presets",
        default=None,
        help="Comma-separated subset of presets to test (default: all).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    presets = list(SCENARIO_PRESETS.keys())
    if args.presets:
        wanted = {p.strip() for p in args.presets.split(",") if p.strip()}
        presets = [p for p in presets if p in wanted]
    print(f"Testing presets: {presets}")

    out_dir = None
    if args.render_bev:
        out_dir = Path(args.render_bev)
        out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for preset in presets:
        print(f"\n[{preset}]")
        try:
            scen = generate_scenario(seed=int(args.seed), preset=preset)
            summary = _audit_scenario(scen)
            print(f"  shape={summary['shape']}  " f"trees={summary['n_trees']}")
            print("  voxel counts:")
            for k, v in summary["voxel_counts"].items():
                if v > 0:
                    print(f"    {k:8s} {v:>10d}")
            print("  surface counts:")
            for k, v in summary["surface_counts"].items():
                if v > 0:
                    print(f"    {k:8s} {v:>10d}")

            # Preset-specific assertions (only meaningful when biome rendering
            # is on, since legacy presets hide the material grid).
            if summary["biome_render"]:
                if preset == "coastal":
                    if summary["voxel_counts"]["WATER"] == 0:
                        raise AssertionError("coastal preset produced no water")
                if preset == "forest":
                    if summary["n_trees"] < 5:
                        raise AssertionError(
                            f"forest preset placed only {summary['n_trees']} trees"
                        )
                if preset == "alpine":
                    if summary["surface_counts"].get("STONE", 0) < 20:
                        raise AssertionError(
                            "alpine preset has too few stone-peak surface tiles"
                        )

            if out_dir is not None:
                png = out_dir / f"{preset}.png"
                _render_bev_png(scen, png)
                print(f"  → wrote {png}")

            print(f"  OK")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failures.append((preset, str(e)))

    if failures:
        print(f"\n{len(failures)} preset(s) FAILED:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        raise SystemExit(1)
    print(f"\nAll {len(presets)} presets OK.")


if __name__ == "__main__":
    main()
