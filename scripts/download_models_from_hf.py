#!/usr/bin/env python
"""Download and smoke-test voxel_car models from Hugging Face Hub.

Example
-------
    python scripts/download_models_from_hf.py \
        --repo-id YOUR_USERNAME/SynthOccPred
"""

from __future__ import annotations

import argparse
import json

from voxel_car.hub import (
    DEFAULT_OCC_FILENAME,
    DEFAULT_RL_FILENAME,
    load_occnet_from_hf,
    load_ppo_from_hf,
)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--occ-file", default=DEFAULT_OCC_FILENAME)
    p.add_argument("--ppo-file", default=DEFAULT_RL_FILENAME)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--token", default=None)
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    model, ego_cfg, cam_cfg, image_hw, device = load_occnet_from_hf(
        repo_id=args.repo_id,
        filename=args.occ_file,
        cache_dir=args.cache_dir,
        revision=args.revision,
        token=args.token,
    )

    ppo = load_ppo_from_hf(
        repo_id=args.repo_id,
        filename=args.ppo_file,
        cache_dir=args.cache_dir,
        revision=args.revision,
        token=args.token,
        device=args.device,
    )

    report = {
        "repo_id": args.repo_id,
        "occupancy": {
            "loaded": True,
            "ego_shape": list(ego_cfg.shape),
            "camera": cam_cfg.to_dict(),
            "image_hw": list(image_hw),
            "device": str(device),
            "n_params": int(model.num_params(model)),
        },
        "ppo": {
            "loaded": True,
            "policy_class": type(ppo.policy).__name__,
            "observation_space": str(ppo.observation_space),
            "action_space": str(ppo.action_space),
        },
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
