"""Training orchestration: one cached sample set produces one optimizer update."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import torch

from dvoc.data.plan import PairedPlan
from dvoc.models.zimage import load_pipeline, precision_dtype
from dvoc.randomness import seed_process
from dvoc.rewards.registry import RewardModels
from dvoc.training.checkpoint import load_checkpoint, save_checkpoint
from dvoc.training.diagnostics import PhaseTimer, log_probe_difference, run_probe
from dvoc.training.distributed import DistributedContext
from dvoc.training.ema import ExponentialMovingAverage
from dvoc.training.evaluation import evaluate
from dvoc.training.logging import RunLogger, write_json
from dvoc.training.phases import accumulate_gradients, apply_update, build_targets, collect_samples


def train(config):
    config.validate()
    context = DistributedContext.initialize()
    try:
        _train(config, context)
    finally:
        context.close()


def _train(config, context):
    plan = PairedPlan(config, context.world_size)
    output = Path(config.output_dir)
    if context.is_main:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Use a new output_dir for each run, including resumed runs")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "config.json", config.to_dict())
    context.barrier()
    with RunLogger(
        output, context.is_main, tracking=config.tracking, config=config.to_dict(), name=config.name
    ) as logger:
        _run_updates(config, context, plan, output, logger)


def _run_updates(config, context, plan, output, logger):
    seed_process(config.seed, context.rank)
    torch.backends.cuda.matmul.allow_tf32 = config.model.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.model.allow_tf32
    dtype = precision_dtype(config.model.precision)
    pipeline, model, parameters = load_pipeline(config.model, context.device)
    context.broadcast_parameters(parameters)
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    hashes = context.gather_objects(digest.hexdigest())
    if len(set(hashes)) != 1:
        raise RuntimeError("Adapter initialization differs across ranks")
    if context.is_main:
        write_json(
            output / "initialization.json",
            {
                "trainable_sha256": hashes[0],
                "world_size": context.world_size,
                "paired_plan_sha256": plan.sha256,
            },
        )
    opt = config.optimizer
    optimizer = torch.optim.AdamW(
        parameters,
        lr=opt.learning_rate,
        betas=(opt.beta1, opt.beta2),
        weight_decay=opt.weight_decay,
        eps=opt.epsilon,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    ema = ExponentialMovingAverage(parameters, opt.ema_decay)
    rewards = RewardModels(context.device)
    # Build training scorers before restoring RNG, so checkpoint continuation
    # starts after any random initialization performed inside external libraries.
    for name in config.reward.weights:
        rewards.get(name)
    first_update = 0
    if config.resume_from:
        first_update = load_checkpoint(config, context, model, optimizer, scaler, ema, plan.sha256)
    optimizer.zero_grad(set_to_none=True)
    last_saved = -1

    if config.evaluation.at_start:
        evaluate(config, context, pipeline, parameters, ema, rewards, logger, first_update, dtype)
    for update_index in range(first_update, config.total_updates):
        step = update_index + 1
        before = run_probe(config, context, pipeline, rewards, logger, step, "before", dtype)
        timer = PhaseTimer(context)
        with timer.measure("rollout"):
            cache = collect_samples(
                config, context, pipeline, plan, rewards, logger, update_index, dtype
            )
        with timer.measure("target"):
            torch.cuda.empty_cache()
            displacement = build_targets(config, pipeline, rewards, cache)
        with timer.measure("fit"):
            torch.cuda.empty_cache()
            metrics = accumulate_gradients(
                config, context, pipeline, cache, displacement, scaler, update_index, dtype
            )
        with timer.measure("optimizer"):
            metrics.update(apply_update(config, context, parameters, optimizer, scaler, ema, step))
        del cache, displacement
        logger.log(step, {**context.reduce_metrics(metrics), **timer.metrics()})
        after = run_probe(config, context, pipeline, rewards, logger, step, "after", dtype)
        log_probe_difference(logger, step, before, after)
        is_last = step == config.total_updates
        scheduled_eval = config.evaluation.every > 0 and step % config.evaluation.every == 0
        if (scheduled_eval and not is_last) or (is_last and config.evaluation.at_end):
            evaluate(config, context, pipeline, parameters, ema, rewards, logger, step, dtype)
        if config.save_every > 0 and step % config.save_every == 0:
            checkpoint = save_checkpoint(
                config, context, model, parameters, optimizer, scaler, ema, step, plan.sha256
            )
            last_saved = step
            logger.checkpoint(checkpoint, step)
        context.barrier()
    if last_saved != config.total_updates:
        checkpoint = save_checkpoint(
            config,
            context,
            model,
            parameters,
            optimizer,
            scaler,
            ema,
            config.total_updates,
            plan.sha256,
        )
        logger.checkpoint(checkpoint, config.total_updates)
    if context.is_main:
        write_json(
            output / "run_done.json",
            {
                "completed_updates": config.total_updates,
                "world_size": context.world_size,
                "paired_plan_sha256": plan.sha256,
            },
        )
    logging.getLogger("dvoc").info("Completed %d updates", config.total_updates)
