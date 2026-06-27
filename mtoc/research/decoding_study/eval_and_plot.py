#!/usr/bin/env python3
"""
Evaluate the decoding-study variant outputs with CometKIWI22 and plot how
translation quality varies with decoding strategy and parameters.

Pipeline:
  1. Find every <model>__<variant>.translations.jsonl in the study output dir.
  2. Score each with pymarian-eval (CometKIWI22, reference-free QE).
  3. Aggregate mean score per (variant, language pair).
  4. Emit plots: temperature sweep, top_p sweep, beam-size sweep, method comparison.

Usage (after launch-decoding-study.sh finishes):
    python eval_decoding_study.py \
        --outputs-dir /mnt/tg/data/projects/wmt26/llm-mt-dec/outputs/decoding-study/outputs \
        --cache /mnt/tg/data/cache/marian/metric \
        --plots-dir /mnt/tg/data/projects/wmt26/llm-mt-dec/outputs/decoding-study/plots \
        --devices all

Reuses the extract/score helpers from eval_sanity.py.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse extraction + scoring logic from the shared-task sanity checker (mtoc/eval_sanity.py).
# This file lives at mtoc/research/decoding_study/, so mtoc/ is two parents up.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_sanity import extract_translation, run_pymarian  # noqa: E402


def parse_variant(filename_stem: str) -> Tuple[str, str]:
    """Split '<model>__<variant>.translations' -> (model, variant)."""
    stem = filename_stem.replace(".translations", "")
    if "__" in stem:
        model, variant = stem.split("__", 1)
        return model, variant
    return stem, "default"


def score_file(path: Path, cmd: List[str], chunk_size: int) -> List[Tuple[dict, Optional[float]]]:
    rows: List[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    out: List[Tuple[dict, Optional[float]]] = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        pairs = [(r.get("source_doc") or "", extract_translation(r)) for r in chunk]
        scores = run_pymarian(pairs, cmd)
        out.extend(zip(chunk, scores))
    return out


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True, type=Path)
    ap.add_argument("--cache", default="/mnt/tg/data/cache/marian/metric")
    ap.add_argument("--metric-model", default="wmt22-cometkiwi-da")
    ap.add_argument("--plots-dir", required=True, type=Path)
    ap.add_argument("--scores-out", type=Path, default=None)
    ap.add_argument("--chunk-size", type=int, default=2048)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--cpu-threads", "-c", type=int, default=20)
    grp.add_argument("--devices", "-d", nargs="+", default=None)
    args = ap.parse_args()

    cmd = ["pymarian-eval", "--stdin", "-m", args.metric_model, "--cache", args.cache]
    if args.devices:
        devices = args.devices
        if devices == ["all"]:
            try:
                n = int(subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
                ).strip().split()[-1]) + 1
                devices = [str(i) for i in range(n)]
            except Exception:
                devices = ["0"]
        cmd += ["-d"] + devices
    else:
        cmd += ["-c", str(args.cpu_threads)]

    files = sorted(args.outputs_dir.glob("*/*.translations.jsonl"))
    if not files:
        print(f"No variant outputs under {args.outputs_dir}", file=sys.stderr)
        sys.exit(1)

    # variant -> lang_pair -> [scores]
    by_var_lp: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    overall: Dict[str, List[float]] = defaultdict(list)

    scores_fh = open(args.scores_out, "w") if args.scores_out else None

    for f in files:
        model, variant = parse_variant(f.stem)
        print(f"Scoring {variant} ({f.name}) ...", flush=True)
        scored = score_file(f, cmd, args.chunk_size)
        for row, score in scored:
            if score is None:
                continue
            src_lang = row.get("src_lang", "?")
            tgt_lang = row.get("tgt_lang", "?")
            lp = f"{src_lang}->{tgt_lang}"
            by_var_lp[variant][lp].append(score)
            overall[variant].append(score)
            if scores_fh:
                scores_fh.write(json.dumps({
                    "variant": variant, "lang_pair": lp,
                    "doc_id": row.get("doc_id"), "score": score,
                }) + "\n")
        print(f"  {variant}: mean={mean(overall[variant]):.4f}  n={len(overall[variant])}")

    if scores_fh:
        scores_fh.close()

    # --- text summary ---
    print("\n" + "=" * 60)
    print("Mean CometKIWI22 by variant (overall):")
    print("=" * 60)
    for variant in sorted(overall, key=lambda v: -mean(overall[v])):
        print(f"  {variant:<14} {mean(overall[variant]):.4f}  (n={len(overall[variant])})")

    # --- plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plots. "
              "Install with: pip install matplotlib", file=sys.stderr)
        return

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    lang_pairs = sorted({lp for v in by_var_lp.values() for lp in v})

    def lp_means(pred) -> Dict[str, List[Tuple[float, float]]]:
        """For variants matching pred -> {lp: [(x, mean_score)]}."""
        out: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for variant, x in pred:
            if variant not in by_var_lp:
                continue
            for lp in lang_pairs:
                vals = by_var_lp[variant].get(lp, [])
                if vals:
                    out[lp].append((x, mean(vals)))
        for lp in out:
            out[lp].sort()
        return out

    # 1. Temperature sweep
    temp_variants = [("sample_t03", 0.3), ("sample_t05", 0.5), ("sample_t07", 0.7),
                     ("sample_t09", 0.9), ("sample_t11", 1.1)]
    data = lp_means(temp_variants)
    if data:
        plt.figure(figsize=(7, 5))
        for lp, pts in data.items():
            xs, ys = zip(*pts)
            plt.plot(xs, ys, marker="o", label=lp)
        # greedy reference line
        if "greedy" in overall:
            plt.axhline(mean(overall["greedy"]), ls="--", color="grey", label="greedy")
        plt.xlabel("temperature (top_p=0.95)")
        plt.ylabel("CometKIWI22")
        plt.title("Quality vs sampling temperature")
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(args.plots_dir / "temperature_sweep.png", dpi=120, bbox_inches="tight")
        plt.close()

    # 2. top_p sweep
    topp_variants = [("topp_080", 0.80), ("topp_090", 0.90), ("sample_t07", 0.95), ("topp_100", 1.00)]
    data = lp_means(topp_variants)
    if data:
        plt.figure(figsize=(7, 5))
        for lp, pts in data.items():
            xs, ys = zip(*pts)
            plt.plot(xs, ys, marker="s", label=lp)
        plt.xlabel("top_p (temperature=0.7)")
        plt.ylabel("CometKIWI22")
        plt.title("Quality vs top_p")
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(args.plots_dir / "topp_sweep.png", dpi=120, bbox_inches="tight")
        plt.close()

    # 3. beam size sweep
    beam_variants = [("greedy", 1), ("beam2", 2), ("beam4", 4), ("beam8", 8)]
    data = lp_means(beam_variants)
    if data:
        plt.figure(figsize=(7, 5))
        for lp, pts in data.items():
            xs, ys = zip(*pts)
            plt.plot(xs, ys, marker="^", label=lp)
        plt.xlabel("beam size (1 = greedy)")
        plt.ylabel("CometKIWI22")
        plt.title("Quality vs beam size")
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(args.plots_dir / "beam_sweep.png", dpi=120, bbox_inches="tight")
        plt.close()

    # 4. method comparison bar (best of each family)
    families = {
        "greedy": ["greedy"],
        "best_sample": ["sample_t03", "sample_t05", "sample_t07", "sample_t09", "sample_t11"],
        "best_beam": ["beam2", "beam4", "beam8"],
    }
    labels, heights = [], []
    for fam, variants in families.items():
        cands = [(v, mean(overall[v])) for v in variants if overall.get(v)]
        if cands:
            best_v, best_m = max(cands, key=lambda x: x[1])
            labels.append(f"{fam}\n({best_v})")
            heights.append(best_m)
    if labels:
        plt.figure(figsize=(7, 5))
        plt.bar(labels, heights, color=["grey", "tab:blue", "tab:orange"])
        for i, h in enumerate(heights):
            plt.text(i, h, f"{h:.3f}", ha="center", va="bottom")
        plt.ylabel("CometKIWI22")
        plt.title("Best of each decoding family")
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(args.plots_dir / "method_comparison.png", dpi=120, bbox_inches="tight")
        plt.close()

    print(f"\nPlots written to {args.plots_dir}")


if __name__ == "__main__":
    main()
