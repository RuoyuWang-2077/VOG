"""Launch each evaluation stage in its intended Python environment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def evaluate_all(config_path, adapter, output, *, gpus=8, qa_python=None, resume=False):
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Run `dvoc evaluate` directly; it launches torchrun workers itself")
    if gpus < 1:
        raise ValueError("gpus must be positive")
    if not qa_python:
        raise ValueError("Pass --qa-python pointing to the Q-Align/DeQA environment")
    output = Path(output).resolve()
    config_path = str(Path(config_path).resolve())
    shared = ["--config", config_path]
    retry = ["--resume"] if resume else []

    def launch(python, args):
        subprocess.run(
            [
                str(python),
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={gpus}",
                "-m",
                "dvoc",
                *args,
            ],
            check=True,
        )

    launch(
        sys.executable,
        [
            "generate",
            *shared,
            "--adapter",
            str(Path(adapter).resolve()),
            "--output",
            str(output / "generated"),
            *retry,
        ],
    )
    manifest = output / "generated/manifest.json"
    for judge in ("regular", "qalign", "deqa"):
        launch(
            sys.executable if judge == "regular" else qa_python,
            [
                "score",
                *shared,
                "--manifest",
                str(manifest),
                "--output",
                str(output / "scores"),
                "--judge",
                judge,
                *retry,
            ],
        )
    from dvoc.config import load_config
    from dvoc.evaluation.report import summarize

    return summarize(load_config(config_path), manifest, output / "scores", output / "report")
