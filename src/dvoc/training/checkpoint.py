"""Complete checkpoint transactions and strict raw-policy continuation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from dvoc.training.logging import write_json


def capture_rng():
    state = np.random.get_state()
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": {
            "bit_generator": state[0],
            "keys": torch.from_numpy(state[1].copy()),
            "pos": state[2],
            "has_gauss": state[3],
            "cached_gaussian": state[4],
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python_random_state"])
    numpy = state["numpy_random_state"]
    np.random.set_state(
        (
            numpy["bit_generator"],
            numpy["keys"].cpu().numpy().astype(np.uint32),
            numpy["pos"],
            numpy["has_gauss"],
            numpy["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def resolve_checkpoint(path):
    root = Path(path)
    if (root / "checkpoint-complete.json").is_file():
        return root
    candidates = [
        p
        for parent in (root, root / "checkpoints")
        for p in parent.glob("checkpoint-*")
        if p.name.removeprefix("checkpoint-").isdigit()
        and (p / "checkpoint-complete.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No complete checkpoint found at {path}")
    return max(candidates, key=lambda p: int(p.name.rsplit("-", 1)[1]))


def continuation_signature(config):
    """Settings that must remain fixed for continuation of the same experiment."""
    raw = config.to_dict()
    return {
        "method": raw["method"],
        "optimizer": raw["optimizer"],
        "reward": raw["reward"],
        "seed": config.seed,
        "model": {
            k: raw["model"][k]
            for k in (
                "path",
                "resolution",
                "precision",
                "lora_rank",
                "lora_alpha",
                "allow_tf32",
                "gradient_checkpointing",
            )
        },
        "sample_batch_size": config.data.sample_batch_size,
        "batches_per_update": config.data.batches_per_update,
    }


def save_checkpoint(config, context, model, parameters, optimizer, scaler, ema, step, plan_hash):
    root = Path(config.output_dir) / "checkpoints"
    final = root / f"checkpoint-{step}"
    temporary = root / f".checkpoint-{step}.incomplete"
    if context.is_main:
        root.mkdir(parents=True, exist_ok=True)
        if final.exists() or temporary.exists():
            raise FileExistsError(f"Checkpoint destination already exists: {final}")
        temporary.mkdir()
    context.barrier()
    rank_dir = temporary / f"rank-{context.rank}"
    rank_dir.mkdir()
    torch.save(capture_rng(), rank_dir / "rng_state.pt")
    write_json(rank_dir / "trainer_state.json", {"epoch_completed": step, "global_step": step})
    if context.is_main:
        model.save_pretrained(temporary / "raw_lora")
        with ema.apply(parameters):
            model.save_pretrained(temporary / "lora")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scaler.state_dict(), temporary / "scaler.pt")
        torch.save(ema.state_dict(), temporary / "ema.pt")
        write_json(temporary / "trainer_state.json", {"epoch_completed": step, "global_step": step})
    context.barrier()
    if context.is_main:
        write_json(
            temporary / "checkpoint-complete.json",
            {
                "format_version": 2,
                "step": step,
                "world_size": context.world_size,
                "paired_plan_sha256": plan_hash,
                "resume_uses": "raw_lora",
                "signature": continuation_signature(config),
            },
        )
        temporary.rename(final)
    context.barrier()
    return final


def load_checkpoint(config, context, model, optimizer, scaler, ema, plan_hash):
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    root = resolve_checkpoint(config.resume_from)
    metadata = json.loads((root / "checkpoint-complete.json").read_text())
    if metadata["world_size"] != context.world_size:
        raise ValueError("Resume requires the checkpoint's original rank count")
    if metadata["paired_plan_sha256"] != plan_hash:
        raise ValueError("Resume paired plan differs from the checkpoint")
    if "signature" in metadata and metadata["signature"] != continuation_signature(config):
        raise ValueError("Continuation settings differ from the checkpoint")
    required = [
        "raw_lora/adapter_config.json",
        "optimizer.pt",
        "scaler.pt",
        "ema.pt",
        "trainer_state.json",
        f"rank-{context.rank}/rng_state.pt",
    ]
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Incomplete resume state: {root / name}")
    position = json.loads((root / "trainer_state.json").read_text())
    step = int(position["global_step"])
    if int(position["epoch_completed"]) != step or metadata["step"] != step:
        raise ValueError("Checkpoint does not represent one optimizer update per epoch")
    if not 0 <= step < config.total_updates:
        raise ValueError("Resume step must be smaller than total_updates")
    weights = load_peft_weights(str(root / "raw_lora"), device=str(context.device))
    result = set_peft_model_state_dict(model, weights, adapter_name="default")
    if result.unexpected_keys or any("lora_" in name for name in result.missing_keys):
        raise ValueError("Checkpoint adapter keys do not match the current model")
    optimizer.load_state_dict(
        torch.load(root / "optimizer.pt", map_location=context.device, weights_only=True)
    )
    scaler.load_state_dict(torch.load(root / "scaler.pt", map_location="cpu", weights_only=True))
    ema.load_state_dict(torch.load(root / "ema.pt", map_location="cpu", weights_only=True))
    for parameter in model.parameters():
        if parameter.requires_grad and not torch.isfinite(parameter).all():
            raise ValueError("Nonfinite adapter parameters in checkpoint")
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("Nonfinite optimizer state in checkpoint")
    restore_rng(
        torch.load(
            root / f"rank-{context.rank}" / "rng_state.pt", map_location="cpu", weights_only=True
        )
    )
    context.barrier()
    return step
