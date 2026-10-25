"""Generate one PNG per fixed prompt/seed from a supplied DVOC adapter."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dvoc.data.plan import read_prompts
from dvoc.evaluation.manifest import sha256, verify_image
from dvoc.models.zimage import (
    autocast,
    encode_prompts,
    load_pipeline,
    precision_dtype,
    rollout,
    sample_noise,
)
from dvoc.randomness import seed_process
from dvoc.training.distributed import DistributedContext
from dvoc.training.logging import write_json


def generate(config, adapter, output, resume=False):
    context = DistributedContext.initialize()
    try:
        generate_on_context(config, context, adapter, output, resume)
    finally:
        context.close()


def generate_on_context(config, context, adapter, output, resume=False):
    adapter, output = Path(adapter).resolve(), Path(output).resolve()
    adapter_files = sorted(adapter.glob("adapter_model.*"))
    if not adapter_files or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError("Provide the checkpoint's lora/ adapter directory")
    prompts = read_prompts(config.data.test_prompts)[: config.benchmark.num_images]
    if len(prompts) != config.benchmark.num_images:
        raise ValueError("The test file contains fewer prompts than benchmark.num_images")
    protocol = {
        "method": "dvoc",
        "count": len(prompts),
        "seed": config.benchmark.seed,
        "resolution": config.model.resolution,
        "steps": config.method.sampling_steps,
        "guidance_scale": 0.0,
        "precision": config.model.precision,
        "allow_tf32": config.model.allow_tf32,
        "model": config.model.path,
        "prompts_sha256": sha256(config.data.test_prompts),
        "adapter_sha256": {path.name: sha256(path) for path in adapter_files},
        "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
        "png_conversion": "clip(image * 255, 0, 255).astype(uint8)",
    }
    protocol_path = output / "generation.json"
    if context.is_main:
        if protocol_path.exists():
            if not resume or json.loads(protocol_path.read_text()) != protocol:
                raise ValueError("Existing generation requires --resume and an identical protocol")
        else:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError("Generation output must be new or contain generation.json")
            write_json(protocol_path, protocol)
        (output / "images").mkdir(exist_ok=True)
    context.barrier()
    seed_process(config.benchmark.seed, context.rank)
    torch.backends.cuda.matmul.allow_tf32 = config.model.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.model.allow_tf32
    settings = copy.deepcopy(config.model)
    settings.initial_adapter = str(adapter)
    settings.gradient_checkpointing = False
    pipeline, model, _ = load_pipeline(settings, context.device)
    model.requires_grad_(False)
    model.eval()
    dtype = precision_dtype(settings.precision)
    records = []
    for index in range(context.rank, len(prompts), context.world_size):
        path = output / "images" / f"{index:05d}.png"
        metadata = path.with_suffix(".json")
        seed = config.benchmark.seed + index
        if metadata.exists():
            row = json.loads(metadata.read_text())
            if (
                not resume
                or row["index"] != index
                or row["prompt"] != prompts[index]
                or row["seed"] != seed
            ):
                raise ValueError(f"Existing sample conflicts with generation: {metadata}")
            verify_image(output / "manifest.json", row)
            records.append(row)
            continue
        with torch.no_grad():
            embeddings = encode_prompts(pipeline, [prompts[index]], context.device)
            noise = sample_noise(pipeline, embeddings, [seed], settings.resolution, context.device)
            with autocast(context.device, dtype):
                result = rollout(pipeline, embeddings, noise, steps=config.method.sampling_steps)
            pixels = result.images[0].float().permute(1, 2, 0).cpu().numpy()
            if not np.isfinite(pixels).all():
                raise FloatingPointError("Nonfinite generated pixels")
            temporary = path.with_suffix(".png.tmp")
            Image.fromarray(np.clip(pixels * 255, 0, 255).astype(np.uint8)).save(
                temporary, format="PNG"
            )
            temporary.replace(path)
            row = {
                "index": index,
                "prompt": prompts[index],
                "seed": seed,
                "image": str(path.relative_to(output)),
                "image_sha256": sha256(path),
                "initial_latent_sha256": hashlib.sha256(
                    noise.detach().float().cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
            }
            write_json(metadata, row)
            records.append(row)
    all_rows = sorted(
        [row for group in context.gather_objects(records) for row in group],
        key=lambda row: row["index"],
    )
    if [r["index"] for r in all_rows] != list(range(len(prompts))):
        raise ValueError("Generation did not produce exactly one image per prompt")
    if context.is_main:
        write_json(output / "manifest.json", {"protocol": protocol, "images": all_rows})
    context.barrier()
