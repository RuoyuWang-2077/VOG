"""Lazy reward loading with distinct differentiable and metric interfaces."""

from __future__ import annotations

import torch


class RewardModels:
    """Share frozen scorers between rollout logging, target ascent, and evaluation."""

    def __init__(self, device):
        self.device = device
        self.models = {}

    def get(self, name):
        if name not in self.models:
            self.models[name] = self._load(name)
        return self.models[name]

    def _load(self, name):
        if name == "clipscore":
            from dvoc.rewards.clip import ClipScorer

            scorer = ClipScorer(self.device)
        elif name == "hpsv3pp":
            from dvoc.rewards.hpsv3pp import HPSv3PPScorer

            scorer = HPSv3PPScorer(device=self.device, iter_norm=0.0)
        elif name == "pickscore":
            from dvoc.rewards.pickscore import PickScoreScorer

            scorer = PickScoreScorer(device=self.device, dtype=torch.float32)
        elif name == "aesthetic":
            from dvoc.rewards.aesthetic import AestheticScorer

            scorer = AestheticScorer(device=self.device, dtype=torch.float32)
        elif name == "imagereward":
            from dvoc.rewards.imagereward import ImageRewardScorer

            scorer = ImageRewardScorer(device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported reward: {name}")
        scorer.requires_grad_(False)
        # The HPS adapter intentionally retains its checkpointed training-mode
        # reward head. The caller controls dropout with an explicit random stream.
        return scorer

    def differentiable(self, images, prompts, weights, *, hps_condition):
        total = None
        for name, weight in weights.items():
            scorer = self.get(name)
            if name == "hpsv3pp":
                scores = scorer._scores(images, prompts, iter_norm=hps_condition)
            elif name == "clipscore":
                texts = scorer.processor(
                    text=prompts, padding="max_length", truncation=True, return_tensors="pt"
                ).to(self.device)
                output = scorer.model(pixel_values=scorer._process(images).to(self.device), **texts)
                scores = (output.logits_per_image.diagonal() / 100).float()
            else:
                raise ValueError(f"Reward {name} has no training gradient adapter")
            weighted = float(weight) * scores
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("At least one training reward is required")
        return total.float()

    @torch.no_grad()
    def score(self, images, prompts, names, *, hps_condition):
        """Preserve each metric's original image preprocessing and scale."""
        result = {}
        for name in names:
            scorer = self.get(name)
            if name == "hpsv3pp":
                scores = scorer(images, prompts, iter_norm=hps_condition)
            elif name == "aesthetic":
                scores = scorer((images * 255).round().clamp(0, 255).to(torch.uint8))
            elif name == "pickscore":
                from PIL import Image

                pixels = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                scores = scorer(prompts, [Image.fromarray(x) for x in pixels.transpose(0, 2, 3, 1)])
            elif name == "imagereward":
                scores = scorer(prompts, images)
            else:
                scores = scorer(images, prompts)
            scores = scores.detach().float()
            if scores.shape != (len(prompts),) or not torch.isfinite(scores).all():
                raise ValueError(f"Invalid scores from {name}: shape={scores.shape}")
            result[name] = scores
        return result
