"""Construct a bounded reward-improving displacement in clean latent space."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from dvoc.randomness import fixed_rng


def select_anchor(endpoint, states, velocities, sigmas, *, mode, query_sigma):
    """Use the sampled endpoint or a late-state clean-latent prediction."""
    if mode == "endpoint":
        return endpoint.detach().float()
    if mode != "proxy":
        raise ValueError(f"Unknown target anchor: {mode}")
    index = int((sigmas[:-1] - query_sigma).abs().argmin())
    return (states[:, index].float() - sigmas[index] * velocities[:, index].float()).detach()


def reward_ascent(
    anchor: torch.Tensor,
    endpoint: torch.Tensor,
    reward: Callable[[torch.Tensor], torch.Tensor],
    seeds: Sequence[int],
    *,
    radius_fraction: float,
    steps: int,
    step_scale: float = 1.0,
) -> torch.Tensor:
    """Return a detached displacement after J normalized projected ascent steps.

    The radius is rho * ||endpoint||, including when ascent starts at a proxy.
    Each reward evaluation uses its own reproducible dropout realization.
    The caller transports this displacement to the actual sampled endpoint.
    """
    if anchor.ndim != 4 or anchor.shape != endpoint.shape:
        raise ValueError("anchor and endpoint must have the same [B,C,H,W] shape")
    if steps < 1 or len(seeds) < steps:
        raise ValueError("Each ascent step requires a reward seed")
    origin = anchor.detach().float()
    budget = (radius_fraction * endpoint.detach().float().flatten(1).norm(dim=1)).view(-1, 1, 1, 1)
    step_length = step_scale * budget / steps
    current = origin.clone()
    for seed in seeds[:steps]:
        current = current.detach().requires_grad_(True)
        with fixed_rng(int(seed), current.device), torch.enable_grad():
            scores = reward(current)
            (gradient,) = torch.autograd.grad(scores.sum(), current)
        if not torch.isfinite(scores).all() or not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite reward or latent gradient")
        if not torch.all(gradient.flatten(1).norm(dim=1) > 0):
            raise FloatingPointError("Reward gradient is zero for a target sample")
        unit_gradient = gradient / (gradient.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
        current = current.detach() + step_length * unit_gradient
        displacement = current - origin
        length = displacement.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        current = (origin + displacement * torch.clamp(budget / (length + 1e-12), max=1.0)).detach()
    displacement = current - origin
    if not torch.isfinite(displacement).all():
        raise FloatingPointError("Nonfinite target displacement")
    if not (displacement.flatten(1).norm(dim=1) <= budget.flatten() * 1.00001 + 1e-6).all():
        raise FloatingPointError("Target exceeded the trust-region radius")
    return displacement
