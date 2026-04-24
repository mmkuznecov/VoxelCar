"""Hugging Face Hub helpers for voxel_car models.

Expected Hub layout:

    <repo-id>/
        occupancy/ckpt_best.pt
        rl/ppo_voxel_car_final.zip
        model_manifest.json
        README.md

The functions here also accept local paths. If the requested filename exists
locally, it is used directly; otherwise it is downloaded from Hugging Face.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Built-in default for the uploaded model repository. Environment variables can
# still override this without code changes.
DEFAULT_HF_REPO = os.environ.get(
    "VOXEL_CAR_HF_REPO",
    "mmkuznecov/SynthOccPredModels",
)
DEFAULT_OCC_FILENAME = os.environ.get(
    "VOXEL_CAR_OCC_HF_FILE",
    "occupancy/ckpt_best.pt",
)
DEFAULT_RL_FILENAME = os.environ.get(
    "VOXEL_CAR_RL_HF_FILE",
    "rl/ppo_voxel_car_final.zip",
)


def _token_or_env(token: Optional[str] = None) -> Optional[str]:
    """Use explicit token first, else HF_TOKEN from the environment."""
    return token or os.environ.get("HF_TOKEN") or None


def _expand_local_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def resolve_model_file(
    path_or_filename: str | Path,
    *,
    repo_id: Optional[str] = None,
    repo_type: str = "model",
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local path or download a file from Hugging Face Hub.

    Parameters
    ----------
    path_or_filename:
        Either an existing local path, or a Hub filename such as
        ``occupancy/ckpt_best.pt``.
    repo_id:
        Hugging Face repo id, e.g. ``mmkuznecov/SynthOccPredModels``.
    repo_type:
        Usually ``model``.
    revision:
        Optional branch, tag, or commit hash.
    cache_dir:
        Optional HF cache directory.
    token:
        Optional HF token. If omitted, HF_TOKEN env var is used if present.
    local_files_only:
        If True, only use already-cached Hub files.

    Returns
    -------
    pathlib.Path
        Local filesystem path to the resolved file.
    """
    candidate = _expand_local_path(path_or_filename)
    if candidate.exists():
        return candidate.resolve()

    repo_id = repo_id or DEFAULT_HF_REPO
    if not repo_id:
        raise FileNotFoundError(
            f"Local file does not exist and no HF repo_id was provided: {candidate}"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for HF downloads. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    downloaded = hf_hub_download(
        repo_id=str(repo_id),
        filename=str(path_or_filename),
        repo_type=repo_type,
        revision=revision,
        cache_dir=None if cache_dir is None else str(cache_dir),
        token=_token_or_env(token),
        local_files_only=bool(local_files_only),
    )
    return Path(downloaded).resolve()


def load_occnet_from_hf(
    *,
    repo_id: Optional[str] = None,
    filename: str = DEFAULT_OCC_FILENAME,
    device=None,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
    image_h_fallback: int = 136,
    image_w_fallback: int = 180,
):
    """Download/load the trained occupancy checkpoint.

    Returns
    -------
    tuple
        ``(model, ego_cfg, cam_cfg, image_hw, device)`` exactly like
        ``load_model_from_ckpt``.
    """
    from .simulation.closed_loop import load_model_from_ckpt

    repo_id = repo_id or DEFAULT_HF_REPO
    ckpt_path = resolve_model_file(
        filename,
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    return load_model_from_ckpt(
        ckpt_path,
        device=device,
        image_h_fallback=image_h_fallback,
        image_w_fallback=image_w_fallback,
    )


def load_ppo_from_hf(
    *,
    repo_id: Optional[str] = None,
    filename: str = DEFAULT_RL_FILENAME,
    device: str = "cpu",
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
):
    """Download/load a Stable-Baselines3 PPO policy."""
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "stable_baselines3 is required to load the PPO policy."
        ) from exc

    repo_id = repo_id or DEFAULT_HF_REPO
    policy_path = resolve_model_file(
        filename,
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    return PPO.load(str(policy_path), device=device)


def load_rl_policy_adapter_from_hf(
    *,
    repo_id: Optional[str] = None,
    filename: str = DEFAULT_RL_FILENAME,
    step_size: float = 1.0,
    max_turn_deg: float = 15.0,
    min_speed_fraction: float = 0.10,
    deterministic: bool = True,
    max_goal_dist: float = 100.0,
    device: str = "cpu",
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
):
    """Download PPO and wrap it in RLPolicyAdapter."""
    from .rl import RLPolicyAdapter

    ppo = load_ppo_from_hf(
        repo_id=repo_id,
        filename=filename,
        device=device,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )

    return RLPolicyAdapter(
        model=ppo,
        step_size=float(step_size),
        max_turn_deg=float(max_turn_deg),
        min_speed_fraction=float(min_speed_fraction),
        deterministic=bool(deterministic),
        max_goal_dist=float(max_goal_dist),
    )
