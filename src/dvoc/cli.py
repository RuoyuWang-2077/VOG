"""Small command-line interface; model imports happen only when needed."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dvoc.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dvoc", description="DVOC training for Z-Image")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("check", "Validate configuration and paired data without loading models"),
        ("train", "Train or resume a DVOC adapter"),
        ("plan", "Create a new paired sampling plan"),
        ("evaluate", "Generate PNGs, run all seven metrics, and publish a report"),
        ("generate", "Generate the fixed PNG evaluation set"),
        ("score", "Score an existing PNG manifest with one judge group"),
        ("summarize", "Merge all seven metrics and log the final report"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
        if name in {"check", "plan"}:
            command.add_argument("--world-size", type=int, default=8)
        if name == "check":
            command.add_argument(
                "--models", action="store_true", help="Also verify local reward files"
            )
        if name == "plan":
            command.add_argument("--images-per-prompt", type=int, default=12)
            command.add_argument("--output", required=True)
        if name == "train":
            command.add_argument("--resume", help="Complete checkpoint or checkpoint parent")
            command.add_argument("--output", help="Fresh output directory")
        if name in {"evaluate", "generate"}:
            command.add_argument("--adapter", required=True, help="Checkpoint's lora/ directory")
        if name in {"evaluate", "generate", "score", "summarize"}:
            command.add_argument("--output", required=True)
        if name in {"evaluate", "generate", "score"}:
            command.add_argument("--resume", action="store_true")
        if name == "evaluate":
            command.add_argument("--gpus", type=int, default=8)
            command.add_argument("--qa-python", required=True)
        if name in {"score", "summarize"}:
            command.add_argument("--manifest", required=True)
        if name == "score":
            command.add_argument("--judge", choices=["regular", "qalign", "deqa"], required=True)
        if name == "summarize":
            command.add_argument("--scores", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "plan":
        from dvoc.data.plan import create_plan

        create_plan(config, args.world_size, args.images_per_prompt, args.output)
        print(f"Wrote paired plan: {Path(args.output).resolve()}")
    elif args.command == "check":
        from dvoc.data.plan import PairedPlan

        plan = PairedPlan(config, args.world_size)
        if config.diagnostics.probe_manifest:
            rows = json.loads(Path(config.diagnostics.probe_manifest).read_text())["samples"]
            if not rows or any(
                not {"sample_id", "prompt", "noise_seed", "reward_seed"} <= row.keys()
                for row in rows
            ):
                raise ValueError("Malformed probe manifest")
        if (
            config.model.initial_adapter
            and not (Path(config.model.initial_adapter) / "adapter_config.json").is_file()
        ):
            raise FileNotFoundError("initial_adapter does not contain adapter_config.json")
        if args.models and "hpsv3pp" in config.reward.weights:
            from dvoc.rewards import paths

            paths.resolve_hpsv3pp_checkpoint()
            paths.resolve_hpsv3pp_config()
            paths.resolve_qwen3vl_model()
            if not Path(os.environ.get("HPSV3PP_SRC", "/nonexistent")).is_dir():
                raise FileNotFoundError("Set HPSV3PP_SRC to the reward source directory")
        print(
            json.dumps(
                {
                    "valid": True,
                    "name": config.name,
                    "world_size": args.world_size,
                    "updates": config.total_updates,
                    "paired_plan_sha256": plan.sha256,
                    "method": config.to_dict()["method"],
                },
                indent=2,
            )
        )
    elif args.command == "train":
        from dvoc.training.trainer import train

        if args.resume:
            config.resume_from = str(Path(args.resume).resolve())
        if args.output:
            config.output_dir = str(Path(args.output).resolve())
        train(config)
    elif args.command == "evaluate":
        from dvoc.evaluation.runner import evaluate_all

        evaluate_all(
            args.config,
            args.adapter,
            args.output,
            gpus=args.gpus,
            qa_python=args.qa_python,
            resume=args.resume,
        )
    elif args.command == "generate":
        from dvoc.evaluation.generate import generate

        generate(config, args.adapter, args.output, args.resume)
    elif args.command == "score":
        from dvoc.evaluation.score import score_manifest

        score_manifest(config, args.manifest, args.output, args.judge, args.resume)
    elif args.command == "summarize":
        from dvoc.evaluation.report import summarize

        report = summarize(config, args.manifest, args.scores, args.output)
        print(json.dumps(report["metrics"], indent=2))
