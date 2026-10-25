"""EMA evaluation with unique prompt indices and isolated random state."""

from __future__ import annotations

import torch

from dvoc.data.plan import read_prompts
from dvoc.models.zimage import autocast, encode_prompts, rollout, sample_noise
from dvoc.randomness import fixed_rng


@torch.no_grad()
def evaluate(config, context, pipeline, parameters, ema, rewards, logger, step, dtype):
    prompts = read_prompts(config.data.test_prompts)
    indices = list(range(context.rank, len(prompts), context.world_size))
    previous_mode = pipeline.transformer.training
    records = []
    settings = config.evaluation
    with fixed_rng(602609260000 + context.rank, context.device), ema.apply(parameters):
        pipeline.transformer.eval()
        try:
            for start in range(0, len(indices), settings.batch_size):
                selected = indices[start : start + settings.batch_size]
                texts = [prompts[index] for index in selected]
                embeddings = encode_prompts(pipeline, texts, context.device)
                noise = sample_noise(
                    pipeline,
                    embeddings,
                    [config.seed + i for i in selected],
                    config.model.resolution,
                    context.device,
                )
                with autocast(context.device, dtype):
                    result = rollout(
                        pipeline, embeddings, noise, steps=config.method.sampling_steps
                    )
                # Keep each metric batch reproducible, including HPS reward dropout.
                with fixed_rng(702609260000 + selected[0], context.device):
                    scores = rewards.score(
                        result.images, texts, settings.metrics, hps_condition=settings.hps_condition
                    )
                for local, index in enumerate(selected):
                    records.append(
                        {
                            "index": index,
                            "prompt": prompts[index],
                            "scores": {name: float(value[local]) for name, value in scores.items()},
                        }
                    )
                    if index < settings.save_images:
                        from PIL import Image

                        destination = logger.directory / "evaluation" / f"step-{step}"
                        destination.mkdir(parents=True, exist_ok=True)
                        pixels = (result.images[local] * 255).round().clamp(0, 255).to(torch.uint8)
                        Image.fromarray(pixels.cpu().permute(1, 2, 0).numpy()).save(
                            destination / f"{index:05d}.png"
                        )
        finally:
            pipeline.transformer.train(previous_mode)
    records = sorted(
        [r for group in context.gather_objects(records) for r in group], key=lambda r: r["index"]
    )
    if [r["index"] for r in records] != list(range(len(prompts))):
        raise ValueError("Evaluation did not cover each prompt exactly once")
    logger.rows(f"evaluation/step-{step}/scores.jsonl", records)
    preview = records[: settings.save_images]
    logger.images(
        step,
        [
            logger.directory / "evaluation" / f"step-{step}" / f"{r['index']:05d}.png"
            for r in preview
        ],
        [r["prompt"] for r in preview],
    )
    logger.log(
        step,
        {
            f"eval/{name}": sum(r["scores"][name] for r in records) / len(records)
            for name in settings.metrics
        },
    )
    context.barrier()
