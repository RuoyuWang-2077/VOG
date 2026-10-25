"""Validate paired prompts and independent random seeds before loading a model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_prompts(path):
    prompts = Path(path).read_text().splitlines()
    if not prompts or any(not line.strip() for line in prompts):
        raise ValueError(f"Prompt file must contain one nonempty prompt per line: {path}")
    return [line.strip() for line in prompts]


class PairedPlan:
    """A sample's prompt, noise, active states, and target seeds travel together."""

    def __init__(self, config, world_size):
        self.config = config
        self.path = Path(config.data.paired_plan)
        self.payload = json.loads(self.path.read_text())
        self.sha256 = file_hash(self.path)
        self.prompts = read_prompts(config.data.train_prompts)
        if self.payload["world_size"] != world_size:
            raise ValueError(
                f"Paired plan requires {self.payload['world_size']} ranks, got {world_size}"
            )
        if len(self.payload["updates"]) < config.total_updates:
            raise ValueError("Paired plan has fewer updates than requested")
        for split, path in (
            ("train", config.data.train_prompts),
            ("test", config.data.test_prompts),
        ):
            if self.payload[f"{split}_sha256"] != file_hash(path):
                raise ValueError(f"{split} prompts differ from the paired plan")
        self._validate(world_size)

    def _validate(self, world_size):
        data, method = self.config.data, self.config.method
        expected = {
            (rank, batch, slot)
            for rank in range(world_size)
            for batch in range(data.batches_per_update)
            for slot in range(data.sample_batch_size)
        }
        for index, update in enumerate(self.payload["updates"][: self.config.total_updates]):
            rows = update["samples"]
            coordinates = [(r["rank"], r["batch"], r["local_slot"]) for r in rows]
            if len(coordinates) != len(expected) or set(coordinates) != expected:
                raise ValueError(f"Incomplete or duplicated sample slots at update {index + 1}")
            for row in rows:
                active = row["active_indices"]
                if len(active) != method.active_steps or len(set(active)) != len(active):
                    raise ValueError("Plan active-state count differs from method.active_steps")
                if any(i < 0 or i >= method.sampling_steps for i in active):
                    raise ValueError("Active state outside the rollout")
                if len(row["target_seeds"]) < method.ascent_steps:
                    raise ValueError("Insufficient target seeds in plan")
                if not 0 <= row["prompt_index"] < len(self.prompts):
                    raise ValueError("Invalid prompt index in plan")

    def rows(self, update_index, rank):
        rows = [r for r in self.payload["updates"][update_index]["samples"] if r["rank"] == rank]
        return sorted(rows, key=lambda r: (r["batch"], r["local_slot"]))

    def prompts_for(self, rows):
        return [self.prompts[row["prompt_index"]] for row in rows]


def fit_permutation(count, update_index, rank, world_size):
    generator = torch.Generator().manual_seed(502609260000 + update_index * world_size + rank)
    return torch.randperm(count, generator=generator).tolist()


def create_plan(config, world_size, images_per_prompt, destination):
    """Create a new experiment plan; the shipped plan preserves the recorded runs."""
    prompts = read_prompts(config.data.train_prompts)
    batch_size = config.data.sample_batch_size
    global_batch = batch_size * world_size
    if images_per_prompt < 1 or global_batch % images_per_prompt:
        raise ValueError("images_per_prompt must divide sample_batch_size * world_size")
    groups = global_batch // images_per_prompt
    if groups > len(prompts):
        raise ValueError("Not enough distinct prompts for one global batch")
    updates = []
    generator = torch.Generator().manual_seed(config.seed)
    sample_id = 0
    for update in range(config.total_updates):
        rows = []
        for batch in range(config.data.batches_per_update):
            selected = torch.randperm(len(prompts), generator=generator)[:groups].tolist()
            repeated = [index for index in selected for _ in range(images_per_prompt)]
            order = torch.randperm(global_batch, generator=generator).tolist()
            for slot, shuffled_index in enumerate(order):
                active = sorted(
                    torch.randperm(config.method.sampling_steps, generator=generator)[
                        : config.method.active_steps
                    ].tolist()
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "batch": batch,
                        "rank": slot // batch_size,
                        "local_slot": slot % batch_size,
                        "prompt_index": repeated[shuffled_index],
                        "active_indices": active,
                        "noise_seed": 202609260000 + sample_id,
                        "target_seeds": [
                            302609260000 + sample_id * max(2, config.method.ascent_steps) + j
                            for j in range(max(2, config.method.ascent_steps))
                        ],
                    }
                )
                sample_id += 1
        updates.append({"update": update + 1, "samples": rows})
    payload = {
        "version": 2,
        "seed": config.seed,
        "world_size": world_size,
        "updates": updates,
        "train_sha256": file_hash(config.data.train_prompts),
        "test_sha256": file_hash(config.data.test_prompts),
    }
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Plan already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
