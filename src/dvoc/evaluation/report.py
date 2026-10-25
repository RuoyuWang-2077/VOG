"""Merge verified judges into a per-image CSV, statistics, and W&B tables."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from dvoc.evaluation.manifest import (
    METRICS,
    PAPER_METRICS,
    read_manifest,
    read_score_rows,
    sha256,
    validate_scores,
    verify_image,
)
from dvoc.training.logging import RunLogger, write_json


def summarize(config, manifest_path, scores_root, output):
    manifest_path, scores_root, output = (
        Path(manifest_path).resolve(),
        Path(scores_root),
        Path(output),
    )
    _, images = read_manifest(manifest_path)
    manifest_hash = sha256(manifest_path)
    merged = {
        row["index"]: {"index": row["index"], "prompt": row["prompt"], "seed": row["seed"]}
        for row in images
    }
    for row in images:
        verify_image(manifest_path, row)
    protocols = {}
    for judge in METRICS:
        target = scores_root / judge
        complete = json.loads((target / "complete.json").read_text())
        protocol = complete["protocol"]
        if protocol["manifest_sha256"] != manifest_hash or complete["count"] != len(images):
            raise ValueError(f"{judge} scored a different or incomplete manifest")
        if (
            protocol["metrics"] != METRICS[judge]
            or protocol["hps_condition"] != config.benchmark.hps_condition
        ):
            raise ValueError(f"{judge} uses a different metric protocol")
        records = [
            r
            for rank in range(protocol["world_size"])
            for r in read_score_rows(target / f"scores-rank{rank}.jsonl")
        ]
        if sorted(r["index"] for r in records) != list(range(len(images))):
            raise ValueError(f"{judge} has incomplete or duplicated samples")
        for row in records:
            if row["image_sha256"] != images[row["index"]]["image_sha256"]:
                raise ValueError("Scored image hash differs from the generated PNG")
            validate_scores(row["scores"], judge)
            merged[row["index"]].update(row["scores"])
        protocols[judge] = protocol
    stats = {}
    for metric in PAPER_METRICS:
        values = [row[metric] for row in merged.values()]
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[metric] = {
            "mean": statistics.mean(values),
            "std": std,
            "stderr": std / len(values) ** 0.5,
            "count": len(values),
        }
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "dvoc",
        "count": len(images),
        "manifest_sha256": manifest_hash,
        "metrics": stats,
        "protocols": protocols,
    }
    write_json(output / "summary.json", report)
    with (output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "prompt", "seed", *PAPER_METRICS])
        writer.writeheader()
        writer.writerows(merged.values())
    with RunLogger(
        output,
        True,
        tracking=config.tracking,
        config=config.to_dict(),
        name=config.name + "-evaluation",
        job_type="evaluation",
    ) as logger:
        means = {f"benchmark/{name}": values["mean"] for name, values in stats.items()}
        logger.log(0, means)
        logger.summary(
            {**means, "benchmark/count": len(images), "benchmark/manifest_sha256": manifest_hash}
        )
        logger.table(
            "benchmark/metrics",
            ["metric", "mean", "std", "stderr", "count"],
            [
                [key, value["mean"], value["std"], value["stderr"], value["count"]]
                for key, value in stats.items()
            ],
        )
        logger.table(
            "benchmark/per_image",
            ["index", "prompt", *PAPER_METRICS],
            [
                [r["index"], r["prompt"], *[r[name] for name in PAPER_METRICS]]
                for r in merged.values()
            ],
        )
        preview = images[: config.benchmark.preview_images]
        logger.images(
            0,
            [manifest_path.parent / row["image"] for row in preview],
            [row["prompt"] for row in preview],
            key="benchmark/images",
        )
    return report
