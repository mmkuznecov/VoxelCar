#!/usr/bin/env python
"""Upload trained voxel_car models to Hugging Face Hub.

Uploads:
    occupancy checkpoint:
        runs/20260419_223836/ckpt_best.pt
        -> occupancy/ckpt_best.pt

    PPO policy:
        rl_runs/easy_oracle_ppo/ppo_voxel_car_final.zip
        -> rl/ppo_voxel_car_final.zip

Example
-------
    huggingface-cli login

    python scripts/upload_models_to_hf.py \
        --repo-id YOUR_USERNAME/SynthOccPred \
        --occ-ckpt runs/20260419_223836/ckpt_best.pt \
        --ppo-model rl_runs/easy_oracle_ppo/ppo_voxel_car_final.zip

Private repo:
    python scripts/upload_models_to_hf.py \
        --repo-id YOUR_USERNAME/SynthOccPred \
        --private
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

OCC_REPO_PATH = "occupancy/ckpt_best.pt"
PPO_REPO_PATH = "rl/ppo_voxel_car_final.zip"
MANIFEST_REPO_PATH = "model_manifest.json"
README_REPO_PATH = "README.md"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--repo-id",
        required=True,
        help="HF model repo id, e.g. YOUR_USERNAME/SynthOccPred.",
    )
    p.add_argument(
        "--occ-ckpt",
        default="runs/20260419_223836/ckpt_best.pt",
        help="Local OccNet checkpoint path.",
    )
    p.add_argument(
        "--ppo-model",
        default="rl_runs/easy_oracle_ppo/ppo_voxel_car_final.zip",
        help="Local Stable-Baselines3 PPO .zip path.",
    )
    p.add_argument(
        "--rl-args",
        default="rl_runs/easy_oracle_ppo/args.json",
        help="Optional RL args.json to upload if present.",
    )
    p.add_argument(
        "--occ-eval-report",
        default="runs/20260419_223836/eval_report.json",
        help="Optional occupancy eval_report.json to upload if present.",
    )

    p.add_argument(
        "--private",
        action="store_true",
        help="Create the HF repo as private.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, uses HF_TOKEN env var or cached login.",
    )
    p.add_argument(
        "--commit-message",
        default="Upload voxel_car occupancy and PPO models",
    )
    p.add_argument(
        "--no-readme",
        action="store_true",
        help="Do not generate/upload README.md.",
    )

    return p.parse_args(argv)


def _token_or_env(token: str | None) -> str | None:
    return token or os.environ.get("HF_TOKEN") or None


def _require_file(path: str | Path, label: str) -> Path:
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        raise SystemExit(f"{label} not found: {p}")
    return p.resolve()


def _optional_file(path: str | Path) -> Path | None:
    p = Path(path).expanduser()
    if p.exists() and p.is_file():
        return p.resolve()
    return None


def _build_manifest(
    *,
    repo_id: str,
    occ_ckpt: Path,
    ppo_model: Path,
    rl_args: Path | None,
    occ_eval_report: Path | None,
) -> dict:
    return {
        "library_name": "voxel_car",
        "repo_id": repo_id,
        "files": {
            "occupancy_checkpoint": OCC_REPO_PATH,
            "ppo_policy": PPO_REPO_PATH,
            "rl_args": "rl/args.json" if rl_args else None,
            "occupancy_eval_report": (
                "occupancy/eval_report.json" if occ_eval_report else None
            ),
        },
        "source_files": {
            "occupancy_checkpoint": str(occ_ckpt),
            "ppo_policy": str(ppo_model),
            "rl_args": str(rl_args) if rl_args else None,
            "occupancy_eval_report": str(occ_eval_report) if occ_eval_report else None,
        },
        "usage": {
            "occ_checkpoint_hf_file": OCC_REPO_PATH,
            "ppo_policy_hf_file": PPO_REPO_PATH,
            "environment_variables": {
                "VOXEL_CAR_HF_REPO": repo_id,
                "VOXEL_CAR_OCC_HF_FILE": OCC_REPO_PATH,
                "VOXEL_CAR_RL_HF_FILE": PPO_REPO_PATH,
            },
        },
    }


def _build_readme(repo_id: str) -> str:
    return f"""---
license: mit
tags:
- occupancy-prediction
- reinforcement-learning
- autonomous-driving
- stable-baselines3
- gradio
library_name: voxel_car
---

# SynthOccPred / voxel_car models

This repository contains two trained artifacts for the `voxel_car` project:

| Artifact | Hub path |
|---|---|
| Occupancy predictor checkpoint | `{OCC_REPO_PATH}` |
| PPO policy for RL planner | `{PPO_REPO_PATH}` |

## Use from Python

```python
from voxel_car.hub import load_occnet_from_hf, load_ppo_from_hf

model, ego_cfg, cam_cfg, image_hw, device = load_occnet_from_hf(
    repo_id="{repo_id}",
)

ppo = load_ppo_from_hf(
    repo_id="{repo_id}",
)
```

## Use in the Gradio demo

```bash
export VOXEL_CAR_HF_REPO="{repo_id}"
python app.py
```

The demo can run closed-loop scenarios with either:

1. standard A* planner over OccNet occupancy predictions
2. PPO-RL planner over OccNet occupancy predictions
"""


def main(argv=None):
    args = _parse_args(argv)

    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with:\n"
            "    pip install huggingface_hub"
        ) from exc

    token = _token_or_env(args.token)

    occ_ckpt = _require_file(args.occ_ckpt, "OccNet checkpoint")
    ppo_model = _require_file(args.ppo_model, "PPO model")
    rl_args = _optional_file(args.rl_args)
    occ_eval_report = _optional_file(args.occ_eval_report)

    print(f"Creating/using HF repo: {args.repo_id}")
    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=bool(args.private),
        exist_ok=True,
        token=token,
    )

    upload_jobs: list[tuple[Path, str]] = [
        (occ_ckpt, OCC_REPO_PATH),
        (ppo_model, PPO_REPO_PATH),
    ]

    if rl_args is not None:
        upload_jobs.append((rl_args, "rl/args.json"))

    if occ_eval_report is not None:
        upload_jobs.append((occ_eval_report, "occupancy/eval_report.json"))

    with tempfile.TemporaryDirectory(prefix="voxel_car_hf_") as td:
        td_path = Path(td)

        manifest = _build_manifest(
            repo_id=args.repo_id,
            occ_ckpt=occ_ckpt,
            ppo_model=ppo_model,
            rl_args=rl_args,
            occ_eval_report=occ_eval_report,
        )
        manifest_path = td_path / "model_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        upload_jobs.append((manifest_path, MANIFEST_REPO_PATH))

        if not args.no_readme:
            readme_path = td_path / "README.md"
            readme_path.write_text(_build_readme(args.repo_id))
            upload_jobs.append((readme_path, README_REPO_PATH))

        for local_path, path_in_repo in upload_jobs:
            print(f"Uploading {local_path} -> {path_in_repo}")
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type="model",
                token=token,
                commit_message=args.commit_message,
            )

    print()
    print("Done.")
    print(f"Repo: https://huggingface.co/{args.repo_id}")
    print()
    print("Recommended environment variables:")
    print(f'export VOXEL_CAR_HF_REPO="{args.repo_id}"')
    print(f'export VOXEL_CAR_OCC_HF_FILE="{OCC_REPO_PATH}"')
    print(f'export VOXEL_CAR_RL_HF_FILE="{PPO_REPO_PATH}"')


if __name__ == "__main__":
    main()
