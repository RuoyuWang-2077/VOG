"""Append-only JSONL records and optional rank-zero Weights & Biases logging."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


class RunLogger:
    def __init__(
        self, directory, enabled, *, tracking=None, config=None, name=None, job_type="train"
    ):
        self.directory = Path(directory)
        self.enabled = enabled
        self.run = None
        self.tracking = tracking
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=True)
            if tracking is not None and tracking.backend == "wandb":
                import wandb

                self.run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", tracking.project),
                    entity=tracking.entity or None,
                    group=tracking.group or None,
                    name=name,
                    tags=tracking.tags,
                    config=config,
                    job_type=job_type,
                    mode=os.environ.get("WANDB_MODE", tracking.mode),
                    dir=str(self.directory),
                    save_code=False,
                )
                self.run.define_metric("update")
                self.run.define_metric("*", step_metric="update")

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        self.finish(exit_code=1 if error_type else 0)

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
            self.run = None

    def log(self, step, metrics):
        if not self.enabled:
            return
        values = {key: float(value) for key, value in metrics.items()}
        with (self.directory / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": step, **values}, allow_nan=False) + "\n")
        logging.getLogger("dvoc").info("update=%s %s", step, values)
        if self.run is not None:
            # Several phases can log at the same optimizer update. A custom axis
            # preserves every record without relying on W&B's monotonic row step.
            self.run.log({"update": step, **values})

    def rows(self, relative_path, rows):
        if self.enabled:
            path = self.directory / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def images(self, step, paths, captions, key="eval/images"):
        if self.run is not None:
            import wandb

            self.run.log(
                {
                    "update": step,
                    key: [
                        wandb.Image(str(path), caption=caption)
                        for path, caption in zip(paths, captions, strict=True)
                    ],
                }
            )

    def table(self, key, columns, rows):
        if self.run is not None:
            import wandb

            self.run.log({key: wandb.Table(columns=columns, data=rows)})

    def summary(self, values):
        if self.run is not None:
            self.run.summary.update(values)

    def checkpoint(self, path, step):
        if self.run is not None:
            self.run.summary.update({"checkpoint/latest": str(path), "checkpoint/update": step})
            if self.tracking.upload_checkpoints:
                import wandb

                artifact = wandb.Artifact(
                    f"dvoc-{self.run.id}-checkpoint", type="model", metadata={"update": step}
                )
                artifact.add_dir(str(path))
                self.run.log_artifact(artifact, aliases=["latest", f"update-{step}"])
