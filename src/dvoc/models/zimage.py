"""Z-Image's list-based transformer, native time convention, and Euler rollout."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]


def precision_dtype(precision: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def autocast(device, dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def load_pipeline(config, device):
    """Freeze the base model and attach one trainable LoRA adapter."""
    from diffusers import ZImagePipeline
    from peft import LoraConfig, PeftModel, get_peft_model

    dtype = precision_dtype(config.precision)
    pipe = ZImagePipeline.from_pretrained(config.path, torch_dtype=dtype)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.to(device, dtype=torch.float32)
    pipe.text_encoder.to(device, dtype=dtype)
    pipe.transformer.to(device)
    if config.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
        pipe.vae.enable_gradient_checkpointing()
    if config.initial_adapter:
        model = PeftModel.from_pretrained(
            pipe.transformer, config.initial_adapter, is_trainable=True
        )
    else:
        model = get_peft_model(
            pipe.transformer,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=LORA_TARGETS,
            ),
        )
    model.set_adapter("default")
    # PEFT injects the adapter into the transformer's existing modules. The raw
    # pipeline transformer and the serialization wrapper share these parameters.
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable LoRA parameters were created")
    pipe.set_progress_bar_config(disable=True)
    return pipe, model, parameters


@torch.no_grad()
def encode_prompts(pipe, prompts, device):
    embeddings, _ = pipe.encode_prompt(
        prompt=list(prompts),
        device=device,
        do_classifier_free_guidance=False,
        max_sequence_length=512,
    )
    return [embedding.to(device) for embedding in embeddings]


def raw_velocity(model, latents, model_time, embeddings):
    """Pack [B,C,H,W] into per-sample [C,1,H,W] inputs, then unpack."""
    try:
        dtype = model.dtype
    except AttributeError:
        dtype = next(model.parameters()).dtype
    packed = list(latents.to(dtype).unsqueeze(2).unbind(0))
    output = model(packed, model_time, embeddings, return_dict=False)[0]
    output = torch.stack(list(output)) if isinstance(output, (list, tuple)) else output
    if output.ndim == 5 and output.shape[2] == 1:
        output = output.squeeze(2)
    return output.float()


def velocity_at_state(model, latents, scheduler_time, embeddings):
    """Euler velocity is the negative raw output, at the stored scheduler time.

    Reconstructing time from sigma can change BF16 rounding. Reuse the actual
    scheduler timestep from the rollout for every differentiable recomputation.
    """
    model_time = torch.full(
        (latents.shape[0],),
        (1000.0 - float(scheduler_time)) / 1000.0,
        device=latents.device,
        dtype=torch.float32,
    )
    return -raw_velocity(model, latents, model_time, embeddings)


def decode_latents(vae, latents):
    scaled = latents / vae.config.scaling_factor + vae.config.shift_factor
    images = vae.decode(scaled.to(vae.dtype), return_dict=False)[0]
    return (images / 2 + 0.5).clamp(0, 1).float()


def sample_noise(pipe, embeddings, seeds, resolution, device):
    """Give every sample an independent generator, stable across batch grouping."""
    values = []
    for seed in seeds:
        value = pipe.prepare_latents(
            1,
            pipe.transformer.config.in_channels,
            resolution,
            resolution,
            embeddings[0].dtype,
            device,
            torch.Generator(device=device).manual_seed(int(seed)),
        )
        values.append(value[0] if isinstance(value, (tuple, list)) else value)
    return torch.cat(values)


@dataclass
class Trajectory:
    states: torch.Tensor  # [B, N+1, C, H, W]
    velocities: torch.Tensor  # [B, N, C, H, W]
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    images: torch.Tensor | None = None

    @property
    def endpoint(self):
        return self.states[:, -1]


@torch.no_grad()
def rollout(pipe, embeddings, noise, *, steps, decode=True):
    """Cache a deterministic Euler trajectory using the configured scheduler."""
    scheduler = pipe.scheduler
    if getattr(scheduler.config, "use_dynamic_shifting", False):
        raise ValueError("This DVOC recipe requires the static Z-Image sigma schedule")
    if getattr(scheduler.config, "stochastic_sampling", False):
        raise ValueError("This DVOC recipe requires deterministic Euler sampling")
    scheduler.set_timesteps(steps, device=noise.device)
    sigmas = scheduler.sigmas.to(noise.device).float().clone()
    times = scheduler.timesteps.to(noise.device).float().clone()
    if len(times) != steps or len(sigmas) != steps + 1:
        raise ValueError("Unexpected scheduler length")
    state = noise.float()
    states, velocities = [state], []
    for index, timestep in enumerate(times):
        velocity = velocity_at_state(pipe.transformer, state, timestep, embeddings)
        state = state + (sigmas[index + 1] - sigmas[index]) * velocity
        velocities.append(velocity)
        states.append(state)
    return Trajectory(
        states=torch.stack(states, dim=1),
        velocities=torch.stack(velocities, dim=1),
        sigmas=sigmas,
        timesteps=times,
        images=decode_latents(pipe.vae, state) if decode else None,
    )
