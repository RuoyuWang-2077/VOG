"""Portable manifests and content verification shared by all evaluation stages."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

METRICS = {
    "regular": ["hpsv3pp", "clipscore", "aesthetic", "imagereward"],
    "qalign": ["qalign_quality", "qalign_aesthetics"],
    "deqa": ["deqa_score"],
}
PAPER_METRICS = [name for group in METRICS.values() for name in group]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    rows = sorted(manifest["images"], key=lambda row: row["index"])
    if not rows or [r["index"] for r in rows] != list(range(len(rows))):
        raise ValueError("Manifest requires unique, contiguous sample indices")
    for row in rows:
        required = {"index", "prompt", "seed", "image", "image_sha256", "initial_latent_sha256"}
        if not required <= row.keys():
            raise ValueError("Incomplete image record")
    return manifest, rows


def verify_image(manifest_path, row):
    path = Path(manifest_path).resolve().parent / row["image"]
    if sha256(path) != row["image_sha256"]:
        raise ValueError(f"Image content changed: {path}")
    return path


def validate_scores(values, judge):
    if set(values) != set(METRICS[judge]):
        raise ValueError(f"Unexpected metric keys for {judge}")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) for value in values.values()
    ):
        raise ValueError("Metric values must be finite numbers")
    if judge in {"qalign", "deqa"} and any(
        not 0.999 <= value <= 5.001 for value in values.values()
    ):
        raise ValueError("Quality/aesthetic MOS must be within [1, 5]")


def read_score_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"Incomplete final JSONL record; reconcile before resuming: {path}")
    return [json.loads(line) for line in data.splitlines()]
