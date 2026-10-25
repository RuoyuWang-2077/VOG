"""Frozen scorers for the seven-metric PNG protocol, loaded one group at a time."""

from __future__ import annotations

import os
from pathlib import Path


def prepare_qalign(source, destination):
    """Point the released model's tokenizer/processor at its local directory."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    code = source / "modeling_mplug_owl2.py"
    if not code.is_file():
        return source
    old = code.read_text()
    needle = 'from_pretrained("q-future/one-align")'
    if needle not in old:
        return source
    if old.count(needle) != 2:
        raise ValueError("Unrecognized Q-Align resource-loading code; use a verified local adapter")
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        target = destination / path.name
        if path.name == code.name:
            text = old.replace(needle, "from_pretrained(config._name_or_path)")
            if target.exists() and target.read_text() != text:
                raise ValueError("Q-Align local adapter differs from the supplied model source")
            target.write_text(text)
        elif not target.exists():
            target.symlink_to(path, target_is_directory=path.is_dir())
    return destination


def load_judge(name, device, hps_condition, quality_path=None):
    import torch

    if name == "regular":
        import numpy as np

        from dvoc.evaluation.manifest import METRICS
        from dvoc.rewards.registry import RewardModels

        models = RewardModels(device)
        for metric in METRICS[name]:
            models.get(metric)

        def score(image, prompt):
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
            pixels = pixels.unsqueeze(0).to(device).float() / 255
            values = models.score(pixels, [prompt], METRICS[name], hps_condition=hps_condition)
            return {key: float(value.item()) for key, value in values.items()}

        return score
    from transformers import AutoModelForCausalLM

    if name not in {"qalign", "deqa"}:
        raise ValueError(f"Unknown judge: {name}")
    variable = "QALIGN_MODEL_PATH" if name == "qalign" else "DEQA_MODEL_PATH"
    model_path = quality_path or os.environ.get(variable, "")
    if not model_path or not Path(model_path).is_dir():
        raise FileNotFoundError(f"Set {variable} to the local scorer model directory")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map={"": device.index},
        low_cpu_mem_usage=True,
    ).eval()
    model.requires_grad_(False)

    def score(image, prompt):
        if name == "qalign":
            return {
                "qalign_quality": float(
                    model.score([image], task_="quality", input_="image").item()
                ),
                "qalign_aesthetics": float(
                    model.score([image], task_="aesthetics", input_="image").item()
                ),
            }
        return {"deqa_score": float(model.score([image]).item())}

    return score
