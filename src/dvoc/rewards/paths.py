"""Explicit local model paths; Hugging Face IDs remain valid fallback inputs."""

from __future__ import annotations

import os
from pathlib import Path


def hf_snapshot(model_id: str) -> str:
    root = os.environ.get("DVOC_MODEL_ROOT", "")
    if root and (Path(root) / model_id).is_dir():
        return str(Path(root) / model_id)
    return model_id


def required_file(variable: str) -> str:
    value = os.environ.get(variable, "")
    if not value or not Path(value).is_file():
        raise FileNotFoundError(f"Set {variable} to an existing file")
    return value


def resolve_hpsv3pp_checkpoint() -> str:
    return required_file("HPSV3PP_CHECKPOINT")


def resolve_hpsv3pp_config() -> str:
    return required_file("HPSV3PP_CONFIG")


def resolve_qwen3vl_model() -> str:
    value = os.environ.get("HPSV3PP_BACKBONE", "")
    if not value:
        raise ValueError("Set HPSV3PP_BACKBONE to the Qwen3-VL backbone directory or model ID")
    return value


def image_reward_root() -> str:
    return os.environ.get("IMAGEREWARD_ROOT", str(Path(CKPT_PATH) / "ImageReward"))


CKPT_PATH = os.environ.get("REWARD_CKPT_PATH", "models/reward_ckpts")
