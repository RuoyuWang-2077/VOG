"""Synchronized phase timing and optional paired pre/post-update reward probes."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from dvoc.models.zimage import autocast, decode_latents, encode_prompts, rollout, sample_noise
from dvoc.randomness import fixed_rng


class PhaseTimer:
    def __init__(self, context):
        self.context = context
        self.seconds = {}

    @contextmanager
    def measure(self, name):
        self.context.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.context.synchronize()
            self.seconds[name] = time.perf_counter() - start

    def metrics(self):
        values = {f"timing/{name}_seconds": value for name, value in self.seconds.items()}
        values["timing/update_seconds"] = sum(self.seconds.values())
        return self.context.reduce_metrics(values, maximum=True)


@torch.no_grad()
def run_probe(config, context, pipeline, rewards, logger, step, when, dtype):
    if not config.diagnostics.probe_manifest or step not in config.diagnostics.probe_updates:
        return None
    rows = json.loads(Path(config.diagnostics.probe_manifest).read_text())["samples"]
    previous_mode = pipeline.transformer.training
    pipeline.transformer.eval()
    records = []
    try:
        for row in rows[context.rank :: context.world_size]:
            with fixed_rng(row["reward_seed"], context.device):
                embeddings = encode_prompts(pipeline, [row["prompt"]], context.device)
                noise = sample_noise(
                    pipeline,
                    embeddings,
                    [row["noise_seed"]],
                    config.model.resolution,
                    context.device,
                )
                with autocast(context.device, dtype):
                    result = rollout(
                        pipeline,
                        embeddings,
                        noise,
                        steps=config.method.sampling_steps,
                        decode=False,
                    )
                with fixed_rng(row["reward_seed"], context.device):
                    score = rewards.differentiable(
                        decode_latents(pipeline.vae, result.endpoint),
                        [row["prompt"]],
                        config.reward.weights,
                        hps_condition=config.reward.hps_condition,
                    )
                if not torch.isfinite(score).all():
                    raise FloatingPointError("Nonfinite diagnostic reward")
                records.append({"sample_id": row["sample_id"], "reward": float(score.item())})
    finally:
        pipeline.transformer.train(previous_mode)
    records = sorted(
        [r for group in context.gather_objects(records) for r in group],
        key=lambda r: r["sample_id"],
    )
    logger.rows(f"probes/update-{step}-{when}.jsonl", records)
    return records


def log_probe_difference(logger, step, before, after):
    if before is None:
        return
    if [r["sample_id"] for r in before] != [r["sample_id"] for r in after]:
        raise ValueError("Probe sample IDs changed across the optimizer update")
    changes = [a["reward"] - b["reward"] for a, b in zip(after, before, strict=True)]
    logger.log(
        step,
        {
            "probe/reward_gain": sum(changes) / len(changes),
            "probe/positive_fraction": sum(value > 0 for value in changes) / len(changes),
        },
    )
