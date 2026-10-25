"""Sparse Euler composition and endpoint-pinned regression."""

from __future__ import annotations

import torch


def compose_endpoint(initial, sigmas, cached_velocities, active_velocities):
    """Replay all Euler steps; only recomputed active velocities carry gradients."""
    endpoint = initial.detach().float()
    for index, cached in enumerate(cached_velocities):
        delta_sigma = (sigmas[index + 1] - sigmas[index]).to(endpoint.device, torch.float32)
        velocity = active_velocities[index] if index in active_velocities else cached.detach()
        endpoint = endpoint + delta_sigma * velocity.float()
    return endpoint


def endpoint_loss(composed, sampled, displacement, normalization_floor=1e-5):
    """Pin the forward value to the sample while differentiating the composition.

    The detached normalization is the per-sample mean absolute displacement.
    Keeping its calculation in float64 preserves the training arithmetic.
    """
    sampled = sampled.detach().float()
    connected = composed + (sampled - composed).detach()
    target = (sampled + displacement.float()).detach()
    axes = tuple(range(1, connected.ndim))
    with torch.no_grad():
        normalizer = (
            (connected.double() - target.double())
            .abs()
            .mean(dim=axes, keepdim=True)
            .clamp(min=normalization_floor)
        )
    loss = ((connected - target).square() / normalizer).mean()
    metrics = {
        "loss": loss.detach(),
        "composition_error": (composed.detach() - sampled).abs().mean(),
        "normalizer": normalizer.detach().float().mean(),
        "displacement_abs": displacement.detach().abs().mean(),
    }
    return loss, metrics
