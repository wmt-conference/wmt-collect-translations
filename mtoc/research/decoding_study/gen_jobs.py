#!/usr/bin/env python3
"""
Generate one `collect.py run` command per (model, variant) for the decoding study,
so they can be fed to GNU parallel and scheduled across GPU slots — as soon as a
slot frees, the next buffered job starts. This handles the uneven cost of beam
variants gracefully (a slow beam job holds only its own slot).

Each emitted line is a complete, single-line shell command WITHOUT CUDA_VISIBLE_DEVICES
(the parallel runner pins one GPU per slot). Output goes to stdout, one job per line.

ASSUMPTION: every selected model is tensor_parallel_size=1 (one GPU per job). The
generator errors out if a tp>1 model is selected, since the 1-GPU-per-slot runner
cannot place it.

Usage:
    python gen_jobs.py \
        --registry registry.yaml \
        --input  /mnt/tg/.../decoding-study.jsonl \
        --output-dir /mnt/tg/.../outputs \
        --log-dir    /mnt/tg/.../logs \
        --models all --variants all
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]  # mtoc/research/decoding_study -> repo root
COLLECT = "mtoc/collect.py"
PYTHON = "/opt/venv/bin/python"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True, type=Path)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--variants", nargs="+", default=["all"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    reg = yaml.safe_load(args.registry.read_text())
    models = reg.get("models", {})
    if not models:
        print("No models in registry", file=sys.stderr)
        sys.exit(1)

    selected = list(models) if args.models == ["all"] else args.models

    jobs = 0
    for model_key in selected:
        if model_key not in models:
            print(f"Unknown model: {model_key}", file=sys.stderr)
            sys.exit(1)
        cfg = models[model_key]
        tp = int(cfg.get("tensor_parallel_size", 1))
        if tp != 1:
            print(f"ERROR: {model_key} has tp={tp}; the per-slot runner only "
                  f"supports tp=1 jobs.", file=sys.stderr)
            sys.exit(1)

        model_variants = list((cfg.get("variants") or {}).keys())
        variant_list = model_variants if args.variants == ["all"] else args.variants

        for variant in variant_list:
            if variant not in model_variants:
                print(f"ERROR: {model_key} has no variant '{variant}'. "
                      f"Available: {model_variants}", file=sys.stderr)
                sys.exit(1)
            cmd = [
                PYTHON, COLLECT, "run",
                "--input", str(args.input),
                "--model-registry", str(args.registry),
                "--models", model_key,
                "--variants", variant,
                "--output-dir", str(args.output_dir),
                "--log-dir", str(args.log_dir),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            ]
            print(" ".join(shlex.quote(c) for c in cmd))
            jobs += 1

    print(f"# generated {jobs} jobs", file=sys.stderr)


if __name__ == "__main__":
    main()
