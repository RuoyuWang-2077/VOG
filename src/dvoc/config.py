"""Typed configuration, explicit defaults, and early validation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class ModelConfig:
    path: str = "Tongyi-MAI/Z-Image-Turbo"
    resolution: int = 1024
    precision: str = "bf16"
    allow_tf32: bool = True
    gradient_checkpointing: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    initial_adapter: str = ""


@dataclass
class DataConfig:
    train_prompts: str = "data/pickapic/train.txt"
    test_prompts: str = "data/pickapic/test.txt"
    paired_plan: str = "data/paired_plan.json"
    sample_batch_size: int = 3
    batches_per_update: int = 24


@dataclass
class MethodConfig:
    sampling_steps: int = 9
    active_steps: int = 3
    anchor: str = "endpoint"
    ascent_steps: int = 2
    radius_fraction: float = 0.10000000000000003
    step_scale: float = 1.0
    proxy_sigma: float = 0.273
    normalization_floor: float = 1e-5


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-4
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    ema_decay: float = 0.9


@dataclass
class RewardConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {"hpsv3pp": 0.03846, "clipscore": 1.0}
    )
    hps_condition: float = 0.0


@dataclass
class EvaluationConfig:
    every: int = 10
    at_start: bool = True
    at_end: bool = True
    batch_size: int = 1
    hps_condition: float = 0.0
    metrics: list[str] = field(
        default_factory=lambda: ["imagereward", "aesthetic", "pickscore", "hpsv3pp", "clipscore"]
    )
    save_images: int = 8


@dataclass
class DiagnosticsConfig:
    probe_manifest: str = "data/probes.json"
    probe_updates: list[int] = field(default_factory=lambda: [1, 10, 20])


@dataclass
class TrackingConfig:
    backend: str = "wandb"
    project: str = "vog"
    entity: str = ""
    group: str = ""
    mode: str = "offline"
    tags: list[str] = field(default_factory=list)
    upload_checkpoints: bool = False


@dataclass
class BenchmarkConfig:
    num_images: int = 1000
    seed: int = 20260923
    hps_condition: float = 0.3
    preview_images: int = 8


@dataclass
class TrainingConfig:
    name: str = "dvoc_endpoint_j2"
    output_dir: str = "runs/dvoc_endpoint_j2"
    seed: int = 20260923
    total_updates: int = 20
    save_every: int = 10
    resume_from: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        positive = {
            "total_updates": self.total_updates,
            "resolution": self.model.resolution,
            "lora_rank": self.model.lora_rank,
            "lora_alpha": self.model.lora_alpha,
            "sample_batch_size": self.data.sample_batch_size,
            "batches_per_update": self.data.batches_per_update,
            "sampling_steps": self.method.sampling_steps,
            "active_steps": self.method.active_steps,
            "ascent_steps": self.method.ascent_steps,
            "evaluation.batch_size": self.evaluation.batch_size,
        }
        for name, value in positive.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.model.resolution % 16:
            raise ValueError("resolution must be divisible by 16")
        if self.model.precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be bf16, fp16, or fp32")
        if not 1 <= self.method.active_steps <= self.method.sampling_steps:
            raise ValueError("active_steps must be between 1 and sampling_steps")
        if self.method.anchor not in {"endpoint", "proxy"}:
            raise ValueError("anchor must be endpoint or proxy")
        for name in ("radius_fraction", "step_scale", "normalization_floor"):
            value = getattr(self.method, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.method.proxy_sigma <= 1:
            raise ValueError("proxy_sigma must lie in [0, 1]")
        opt = self.optimizer
        for name in ("learning_rate", "epsilon", "max_grad_norm"):
            value = getattr(opt, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"optimizer.{name} must be finite and positive")
        if not math.isfinite(opt.weight_decay) or opt.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if any(not 0 <= value < 1 for value in (opt.beta1, opt.beta2, opt.ema_decay)):
            raise ValueError("optimizer betas and ema_decay must lie in [0, 1)")
        if not self.reward.weights or set(self.reward.weights) - {"hpsv3pp", "clipscore"}:
            raise ValueError("Training rewards supported: hpsv3pp and clipscore")
        if any(not math.isfinite(v) or v <= 0 for v in self.reward.weights.values()):
            raise ValueError("Reward weights must be finite and positive")
        for condition in (self.reward.hps_condition, self.evaluation.hps_condition):
            if not 0 <= condition <= 1:
                raise ValueError("HPS condition must lie in [0, 1]")
        allowed = {"hpsv3pp", "clipscore", "pickscore", "aesthetic", "imagereward"}
        if not self.evaluation.metrics or set(self.evaluation.metrics) - allowed:
            raise ValueError(f"Evaluation metrics must be selected from {sorted(allowed)}")
        for value in (self.save_every, self.evaluation.every, self.evaluation.save_images):
            if type(value) is not int or value < 0:
                raise ValueError("Save/evaluation intervals and image counts must be nonnegative")
        if self.tracking.backend not in {"wandb", "jsonl"}:
            raise ValueError("tracking.backend must be wandb or jsonl")
        if self.tracking.mode not in {"online", "offline", "disabled"}:
            raise ValueError("tracking.mode must be online, offline, or disabled")
        if self.benchmark.num_images < 1 or self.benchmark.preview_images < 0:
            raise ValueError("Invalid benchmark image count")
        if not 0 <= self.benchmark.hps_condition <= 1:
            raise ValueError("benchmark.hps_condition must lie in [0, 1]")


_SECTIONS = {
    "model": ModelConfig,
    "data": DataConfig,
    "method": MethodConfig,
    "optimizer": OptimizerConfig,
    "reward": RewardConfig,
    "evaluation": EvaluationConfig,
    "diagnostics": DiagnosticsConfig,
    "tracking": TrackingConfig,
    "benchmark": BenchmarkConfig,
}


def load_config(path: str | Path) -> TrainingConfig:
    """Read JSON; relative file paths are relative to the project (configs/..)."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    unknown = set(raw) - {f.name for f in fields(TrainingConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for name, cls in _SECTIONS.items():
        try:
            raw[name] = cls(**raw.get(name, {}))
        except TypeError as error:
            raise ValueError(f"Invalid {name} configuration: {error}") from error
    config = TrainingConfig(**raw)
    config.validate()
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    for obj, keys in (
        (config, ("output_dir", "resume_from")),
        (config.data, ("train_prompts", "test_prompts", "paired_plan")),
        (config.model, ("initial_adapter",)),
        (config.diagnostics, ("probe_manifest",)),
    ):
        for key in keys:
            value = getattr(obj, key)
            if value:
                expanded = Path(os.path.expandvars(value)).expanduser()
                setattr(obj, key, str((root / expanded).resolve()))
    config.model.path = os.path.expandvars(os.path.expanduser(config.model.path))
    if config.model.path.startswith("."):
        config.model.path = str((root / config.model.path).resolve())
    return config
