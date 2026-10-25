"""The three training phases: sample trajectories, build targets, fit adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dvoc.algorithm.objective import compose_endpoint, endpoint_loss
from dvoc.algorithm.targets import reward_ascent, select_anchor
from dvoc.data.plan import fit_permutation
from dvoc.models.zimage import (
    autocast,
    decode_latents,
    encode_prompts,
    rollout,
    sample_noise,
    velocity_at_state,
)
from dvoc.randomness import fixed_rng


@dataclass
class SampleCache:
    embeddings: list
    states: torch.Tensor
    velocities: torch.Tensor
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    rows: list[dict]
    prompts: list[str]


def collect_samples(config, context, pipeline, plan, rewards, logger, update_index, dtype):
    """Sample the current policy with no graph; retain its states and velocities."""
    pipeline.transformer.eval()
    rows = plan.rows(update_index, context.rank)
    batch_size = config.data.sample_batch_size
    states, velocities, embeddings, records = [], [], [], []
    schedule = None
    for batch_index, start in enumerate(range(0, len(rows), batch_size)):
        batch_rows = rows[start : start + batch_size]
        prompts = plan.prompts_for(batch_rows)
        batch_embeddings = encode_prompts(pipeline, prompts, context.device)
        noise = sample_noise(
            pipeline,
            batch_embeddings,
            [r["noise_seed"] for r in batch_rows],
            config.model.resolution,
            context.device,
        )
        with autocast(context.device, dtype):
            trajectory = rollout(
                pipeline, batch_embeddings, noise, steps=config.method.sampling_steps
            )
        if schedule is None:
            schedule = (trajectory.sigmas, trajectory.timesteps)
        elif not all(
            torch.equal(a, b)
            for a, b in zip(schedule, (trajectory.sigmas, trajectory.timesteps), strict=True)
        ):
            raise RuntimeError("Scheduler changed between batches in one update")
        seed = 402609260000 + update_index * 1000 + batch_index * context.world_size + context.rank
        with fixed_rng(seed, context.device):
            scores = rewards.score(
                trajectory.images,
                prompts,
                config.reward.weights,
                hps_condition=config.reward.hps_condition,
            )
        for index, (row, prompt) in enumerate(zip(batch_rows, prompts, strict=True)):
            values = {name: float(value[index]) for name, value in scores.items()}
            values["weighted"] = sum(
                config.reward.weights[name] * values[name] for name in config.reward.weights
            )
            records.append({"sample_id": row["sample_id"], "prompt": prompt, "rewards": values})
        states.append(trajectory.states)
        velocities.append(trajectory.velocities)
        embeddings.extend(batch_embeddings)
        del trajectory
    gathered = [row for rank_rows in context.gather_objects(records) for row in rank_rows]
    logger.rows(f"train_rewards/update-{update_index + 1:04d}.jsonl", gathered)
    logger.log(
        update_index,
        {
            f"train/reward_{key}": sum(r["rewards"][key] for r in gathered) / len(gathered)
            for key in gathered[0]["rewards"]
        },
    )
    return SampleCache(
        embeddings,
        torch.cat(states),
        torch.cat(velocities),
        *schedule,
        rows,
        plan.prompts_for(rows),
    )


def build_targets(config, pipeline, rewards, cache):
    """Optimize one detached latent at a time to bound reward-model memory."""
    method = config.method
    endpoint = cache.states[:, -1]
    displacement = torch.zeros_like(endpoint, dtype=torch.float32)
    for index, row in enumerate(cache.rows):
        sample = slice(index, index + 1)
        anchor = select_anchor(
            endpoint[sample],
            cache.states[sample],
            cache.velocities[sample],
            cache.sigmas,
            mode=method.anchor,
            query_sigma=method.proxy_sigma,
        )

        def latent_reward(value):
            images = decode_latents(pipeline.vae, value)
            return rewards.differentiable(
                images,
                [cache.prompts[index]],
                config.reward.weights,
                hps_condition=config.reward.hps_condition,
            )

        displacement[sample] = reward_ascent(
            anchor,
            endpoint[sample],
            latent_reward,
            row["target_seeds"],
            radius_fraction=method.radius_fraction,
            steps=method.ascent_steps,
            step_scale=method.step_scale,
        )
    return displacement


def accumulate_gradients(
    config, context, pipeline, cache, displacement, scaler, update_index, dtype
):
    """Accumulate one sample at a time, using that sample's own K active states."""
    pipeline.transformer.train()
    count = len(cache.rows)
    order = fit_permutation(count, update_index, context.rank, context.world_size)
    totals = {}
    for index in order:
        sample = slice(index, index + 1)
        active = cache.rows[index]["active_indices"]
        active_velocities = {}
        for state_index in active:
            with autocast(context.device, dtype):
                active_velocities[state_index] = velocity_at_state(
                    pipeline.transformer,
                    cache.states[sample, state_index].detach(),
                    cache.timesteps[state_index],
                    [cache.embeddings[index]],
                )
        composed = compose_endpoint(
            cache.states[sample, 0],
            cache.sigmas,
            list(cache.velocities[sample].unbind(1)),
            active_velocities,
        )
        loss, metrics = endpoint_loss(
            composed,
            cache.states[sample, -1],
            displacement[sample],
            config.method.normalization_floor,
        )
        scaler.scale(loss / count).backward()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) / count
    return totals


def apply_update(config, context, parameters, optimizer, scaler, ema, completed_updates):
    """Average LoRA gradients once, clip them, then update AdamW and EMA."""
    before = [parameter.detach().float().clone() for parameter in parameters]
    scaler.unscale_(optimizer)
    context.average_gradients(parameters)
    norm = torch.nn.utils.clip_grad_norm_(parameters, config.optimizer.max_grad_norm)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise FloatingPointError(f"Invalid DVOC adapter gradient norm: {norm}")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    ema.update(parameters, completed_updates)
    displacement = sum(
        (p.detach().float() - b).square().sum() for p, b in zip(parameters, before, strict=True)
    ).sqrt()
    previous_norm = sum(b.square().sum() for b in before).sqrt()
    return {
        "optimizer/gradient_norm": float(norm),
        "optimizer/gradient_clipped": float(norm > config.optimizer.max_grad_norm),
        "optimizer/parameter_displacement": float(displacement),
        "optimizer/relative_parameter_displacement": float(displacement / (previous_norm + 1e-12)),
    }
