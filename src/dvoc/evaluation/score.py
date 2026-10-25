"""Resumable per-image scoring with input hashes and complete coverage checks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from PIL import Image

from dvoc.evaluation.judges import load_judge, prepare_qalign
from dvoc.evaluation.manifest import (
    METRICS,
    read_manifest,
    read_score_rows,
    sha256,
    validate_scores,
    verify_image,
)
from dvoc.randomness import fixed_rng
from dvoc.training.distributed import DistributedContext
from dvoc.training.logging import write_json


def score_manifest(config, manifest_path, output, judge, resume=False):
    context = DistributedContext.initialize()
    try:
        score_on_context(config, context, manifest_path, output, judge, resume)
    finally:
        context.close()


def score_on_context(
    config, context, manifest_path, output, judge, resume=False, scorer_factory=load_judge
):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    _, images = read_manifest(manifest_path)
    target = output / judge
    protocol = {
        "judge": judge,
        "metrics": METRICS[judge],
        "count": len(images),
        "world_size": context.world_size,
        "manifest_sha256": sha256(manifest_path),
        "hps_condition": config.benchmark.hps_condition,
        "batch_size": 1,
        "reward_seed_base": config.benchmark.seed,
        "tf32": judge == "regular",
        "model_path": os.environ.get(
            "QALIGN_MODEL_PATH" if judge == "qalign" else "DEQA_MODEL_PATH", ""
        )
        if judge != "regular"
        else "configured reward models",
    }
    protocol_path = target / "protocol.json"
    if context.is_main:
        if protocol_path.exists():
            if not resume or json.loads(protocol_path.read_text()) != protocol:
                raise ValueError(
                    "Scoring resume requires the identical manifest, rank count, and protocol"
                )
        else:
            if target.exists() and any(target.iterdir()):
                raise FileExistsError("Score output contains files without a protocol")
            write_json(protocol_path, protocol)
        if judge == "qalign":
            source = os.environ.get("QALIGN_MODEL_PATH", "")
            if not source or not Path(source).is_dir():
                raise FileNotFoundError("Set QALIGN_MODEL_PATH to the local Q-Align directory")
            prepared = prepare_qalign(source, target / "local_model")
            write_json(target / "model.json", {"path": str(prepared)})
    context.barrier()
    local_images = [r for r in images if r["index"] % context.world_size == context.rank]
    lookup = {r["index"]: r for r in local_images}
    path = target / f"scores-rank{context.rank}.jsonl"
    previous = read_score_rows(path)
    seen = set()
    for row in previous:
        index = row["index"]
        if (
            index in seen
            or index not in lookup
            or row["image_sha256"] != lookup[index]["image_sha256"]
        ):
            raise ValueError("Existing scores do not match the input manifest")
        validate_scores(row["scores"], judge)
        seen.add(index)
    # Recheck all local images, including already-scored inputs on resume.
    image_paths = {row["index"]: verify_image(manifest_path, row) for row in local_images}
    pending = [row for row in local_images if row["index"] not in seen]
    torch.backends.cuda.matmul.allow_tf32 = judge == "regular"
    torch.backends.cudnn.allow_tf32 = judge == "regular"
    if pending:
        quality_path = (
            json.loads((target / "model.json").read_text())["path"] if judge == "qalign" else None
        )
        scorer = scorer_factory(judge, context.device, config.benchmark.hps_condition, quality_path)
        with path.open("a") as handle:
            for row in pending:
                with Image.open(image_paths[row["index"]]) as source:
                    image = source.convert("RGB")
                    with (
                        torch.no_grad(),
                        fixed_rng(config.benchmark.seed + row["index"], context.device),
                    ):
                        values = scorer(image, row["prompt"])
                validate_scores(values, judge)
                if any(value == -10 for value in values.values()):
                    raise ValueError("Scorer returned an invalid-score sentinel")
                handle.write(
                    json.dumps(
                        {
                            "index": row["index"],
                            "image_sha256": row["image_sha256"],
                            "scores": values,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                handle.flush()
    elif not path.exists():
        path.touch()
    context.barrier()
    if context.is_main:
        all_scores = [
            r
            for rank in range(context.world_size)
            for r in read_score_rows(target / f"scores-rank{rank}.jsonl")
        ]
        if sorted(r["index"] for r in all_scores) != list(range(len(images))):
            raise ValueError("Incomplete or duplicated score coverage")
        write_json(target / "complete.json", {"count": len(images), "protocol": protocol})
    context.barrier()
