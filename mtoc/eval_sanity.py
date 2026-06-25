#!/usr/bin/env python3
"""
Sanity-check WMT26 blindset translation outputs with CometKIWI22 (reference-free QE).

Extracts clean translations from raw LLM outputs (strips <think>...</think> blocks
and Harmony/gpt-oss 'final' channels), detects suspected meta-responses, scores
with pymarian-eval, and reports breakdowns by model, language pair, and instruction style.

Usage — CPU (while translation job is running):
    python mtoc/eval_sanity.py \\
        --outputs-dir /mnt/tg/.../outputs/blindset-260624/outputs \\
        --cache /mnt/tg/data/cache/marian/metric \\
        --cpu-threads 20

Usage — GPU (after translation job finishes):
    python mtoc/eval_sanity.py \\
        --outputs-dir /mnt/tg/.../outputs/blindset-260624/outputs \\
        --cache /mnt/tg/data/cache/marian/metric \\
        --devices all

Append --models qwen3_5_9b tower_9b  to score specific models only.
Append --out scores.jsonl             to save per-row scores for further analysis.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Translation extraction
# ---------------------------------------------------------------------------

def extract_translation(row: dict) -> str:
    """Return the clean translation, stripping reasoning/thinking channels."""
    text = (row.get("translation") or "").strip()
    model_key = (row.get("model_key") or "").lower()

    # gpt-oss Harmony format: keep only the 'final' channel that follows
    # the 'assistantfinal' marker. The reasoning analysis before it is discarded.
    if "gpt_oss" in model_key or "gpt-oss" in model_key:
        marker = "assistantfinal"
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
        # Secondary marker variant
        idx = text.rfind("\nfinal")
        if idx != -1 and "analysis" in text[:idx]:
            return text[idx + len("\nfinal"):].strip()

    # Qwen-style thinking: strip <think>...</think>, keep the answer that follows.
    if "</think>" in text:
        idx = text.find("</think>")
        return text[idx + len("</think>"):].strip()

    return text


# ---------------------------------------------------------------------------
# Instruction style classification
# ---------------------------------------------------------------------------

def classify_instruction(instruction: str) -> str:
    """Bucket the instruction into one of three styles."""
    instr = instruction.strip()
    if instr.startswith("You are a professional"):
        return "professional"
    if instr.startswith("Translate from") or instr.startswith("Translate the"):
        return "simple"
    return "special"


# ---------------------------------------------------------------------------
# Suspicious-output detection
# ---------------------------------------------------------------------------

# Phrases that indicate the model described the task instead of translating.
_META_PHRASES = [
    "let me translate",
    "i need to translate",
    "i will translate",
    "i'll translate",
    "here is the translation",
    "here's the translation",
    "the translation is",
    "translation:",
    "translated text:",
    "sure, here",
    "of course,",
    "here's a thinking process",
    "analyze user input",
    "thinking process:",
    "source text:",
    "target language:",
    "step 1:",
    "step 2:",
]

def check_suspicious(source: str, translation: str, instruction: str = "") -> Optional[str]:
    """
    Return a short reason string if the output looks wrong, else None.
    Checks: empty, too short, meta-response (model describes rather than translates),
    and instruction echo (model repeats the instruction verbatim).
    """
    if not translation:
        return "empty"

    if len(translation) < max(5, len(source) * 0.04):
        return "too_short"

    tl_lower = translation.lower()[:300]

    for phrase in _META_PHRASES:
        if phrase in tl_lower:
            return f"meta:{phrase}"

    # If the translation starts with the instruction (model echoed it)
    if instruction and translation.lower().startswith(instruction.lower()[:30]):
        return "echo_instruction"

    return None


# ---------------------------------------------------------------------------
# pymarian-eval scoring
# ---------------------------------------------------------------------------

def run_pymarian(pairs: List[tuple[str, str]], cmd: List[str]) -> List[Optional[float]]:
    """
    Score a list of (source, translation) pairs.
    Returns a list of floats (or None for failed rows) in the same order.
    """
    inp_lines: List[str] = []
    valid_positions: List[int] = []

    for i, (src, tgt) in enumerate(pairs):
        src_clean = src.replace("\t", " ").replace("\n", " ").strip()
        tgt_clean = tgt.replace("\t", " ").replace("\n", " ").strip()
        if src_clean and tgt_clean:
            inp_lines.append(f"{src_clean}\t{tgt_clean}")
            valid_positions.append(i)

    scores: List[Optional[float]] = [None] * len(pairs)
    if not inp_lines:
        return scores

    result = subprocess.run(
        cmd,
        input="\n".join(inp_lines) + "\n",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"  [pymarian-eval error] {result.stderr[:300]}",
            file=sys.stderr,
        )
        return scores

    out_lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    for pos, score_str in zip(valid_positions, out_lines):
        try:
            scores[pos] = float(score_str)
        except ValueError:
            pass
    return scores


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _stats(vals: List[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
    }


def print_table(
    title: str,
    data: Dict[str, List[float]],
    top_n: int = 50,
    sort_by: str = "mean",
) -> None:
    print(f"\n{'='*72}")
    print(title)
    print(f"{'='*72}")
    items = [
        (k, _stats(v)) for k, v in data.items() if v
    ]
    reverse = sort_by != "key"
    items.sort(key=lambda x: x[1].get(sort_by, 0) if sort_by != "key" else x[0], reverse=reverse)
    items = items[:top_n]
    print(f"{'Key':<45} {'N':>7}  {'Mean':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 80)
    for key, s in items:
        print(
            f"{key:<45} {s['n']:>7}  {s['mean']:>8.4f}  {s['min']:>8.4f}  {s['max']:>8.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check MT outputs with CometKIWI22")
    parser.add_argument(
        "--outputs-dir", required=True, type=Path,
        help="Directory containing <model>/<model>.translations.jsonl files",
    )
    parser.add_argument("--cache", default="/mnt/tg/data/cache/marian/metric")
    parser.add_argument("--metric-model", default="wmt22-cometkiwi-da")
    parser.add_argument("--models", nargs="*", default=None, help="Model keys to score (default: all found)")
    parser.add_argument("--out", type=Path, default=None, help="Save per-row results to JSONL")
    parser.add_argument("--chunk-size", type=int, default=4096,
                        help="Rows per pymarian-eval call (default: 4096)")

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--cpu-threads", "-c", type=int, default=20,
                               help="CPU threads for pymarian-eval (default: 20)")
    device_group.add_argument("--devices", "-d", nargs="+", default=None,
                               metavar="DEV",
                               help="GPU device(s): e.g. 0 1 2 or all")
    args = parser.parse_args()

    # --- Build pymarian-eval command ---
    cmd = [
        "pymarian-eval", "--stdin",
        "-m", args.metric_model,
        "--cache", args.cache,
    ]
    if args.devices:
        devices = args.devices
        # Expand "all" to available GPU indices
        if devices == ["all"]:
            try:
                import subprocess as _sp
                n = int(_sp.check_output(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    text=True,
                ).strip().split()[-1]) + 1
                devices = [str(i) for i in range(n)]
            except Exception:
                devices = ["0"]
        cmd += ["-d"] + devices
    else:
        cmd += ["-c", str(args.cpu_threads)]

    print("Command:", " ".join(cmd))

    # --- Discover output files ---
    outputs_dir = args.outputs_dir
    output_files = sorted(outputs_dir.glob("*/*.translations.jsonl"))
    if args.models:
        output_files = [f for f in output_files if f.parent.name in args.models]

    if not output_files:
        print(f"No output files found under {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(output_files)} model output(s):\n  " +
          "\n  ".join(f.parent.name for f in output_files))

    # --- Accumulators ---
    by_model: Dict[str, List[float]] = defaultdict(list)
    by_lang: Dict[str, List[float]] = defaultdict(list)          # model|tgt_lang
    by_instr_style: Dict[str, List[float]] = defaultdict(list)   # model|style
    by_model_lang: Dict[str, List[float]] = defaultdict(list)    # model|lang (for cross-model lang comparison)

    suspicious_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_rows: Dict[str, int] = {}

    out_fh = open(args.out, "w") if args.out else None

    # --- Score each model ---
    for output_file in output_files:
        model_key = output_file.parent.name
        print(f"\n{'─'*60}")
        print(f"Model: {model_key}  ({output_file})")

        rows: list[dict] = []
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        total_rows[model_key] = len(rows)
        print(f"  Rows: {len(rows)}")

        # Quality pre-check
        sus_counts: Dict[str, int] = defaultdict(int)
        for row in rows:
            src = row.get("source_doc") or ""
            tgt = extract_translation(row)
            instr = row.get("instruction") or ""
            reason = check_suspicious(src, tgt, instr)
            if reason:
                sus_counts[reason] += 1
            suspicious_counts[model_key][reason or "ok"] += 1

        if sus_counts:
            total_sus = sum(sus_counts.values())
            pct = 100.0 * total_sus / max(1, len(rows))
            print(f"  Suspicious: {total_sus}/{len(rows)} ({pct:.1f}%): {dict(sus_counts)}")
        else:
            print("  Suspicious: 0 (all clean)")

        # Score in chunks
        all_scores: List[Optional[float]] = []
        for start in range(0, len(rows), args.chunk_size):
            chunk = rows[start: start + args.chunk_size]
            pairs = [
                (row.get("source_doc") or "", extract_translation(row))
                for row in chunk
            ]
            chunk_scores = run_pymarian(pairs, cmd)
            all_scores.extend(chunk_scores)
            valid = [s for s in chunk_scores if s is not None]
            end = min(start + args.chunk_size, len(rows))
            if valid:
                print(
                    f"  [{end:>6}/{len(rows)}]  chunk mean={sum(valid)/len(valid):.4f}"
                    f"  scored={len(valid)}/{len(chunk)}",
                    flush=True,
                )

        # Aggregate
        scored = 0
        for row, score in zip(rows, all_scores):
            if score is None:
                continue
            scored += 1
            tgt_lang = row.get("tgt_lang") or "unknown"
            instr = row.get("instruction") or ""
            style = classify_instruction(instr)

            by_model[model_key].append(score)
            by_lang[f"{model_key}|{tgt_lang}"].append(score)
            by_instr_style[f"{model_key}|{style}"].append(score)
            by_model_lang[tgt_lang].append(score)   # cross-model lang difficulty

            if out_fh:
                out_fh.write(json.dumps({
                    "model_key": model_key,
                    "doc_id": row.get("doc_id"),
                    "tgt_lang": tgt_lang,
                    "instr_style": style,
                    "score": score,
                    "suspicious": check_suspicious(
                        row.get("source_doc") or "", extract_translation(row), instr
                    ),
                }) + "\n")

        valid_scores = by_model[model_key]
        if valid_scores:
            print(
                f"  ✓ Overall mean={sum(valid_scores)/len(valid_scores):.4f}"
                f"  scored={scored}/{len(rows)}"
            )

    if out_fh:
        out_fh.close()
        print(f"\nPer-row scores saved to {args.out}")

    # --- Summary tables ---
    print_table("CometKIWI22 — by model (higher = better)", by_model)

    print_table(
        "CometKIWI22 — by model × instruction style",
        by_instr_style,
        top_n=60,
        sort_by="key",
    )

    # Per-language breakdown: show worst languages per model
    # Group by model and print top+bottom 5 langs for each
    print(f"\n{'='*72}")
    print("CometKIWI22 — worst 8 languages per model")
    print(f"{'='*72}")
    per_model_lang: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for key, scores in by_lang.items():
        model_k, lang = key.split("|", 1)
        per_model_lang[model_k][lang] = scores

    for model_k in sorted(per_model_lang):
        lang_means = {
            lang: sum(vals) / len(vals)
            for lang, vals in per_model_lang[model_k].items()
            if vals
        }
        worst = sorted(lang_means.items(), key=lambda x: x[1])[:8]
        best = sorted(lang_means.items(), key=lambda x: -x[1])[:3]
        print(f"\n  {model_k}")
        print(f"    Best:  " + "  ".join(f"{l}={s:.3f}" for l, s in best))
        print(f"    Worst: " + "  ".join(f"{l}={s:.3f}" for l, s in worst))

    print(f"\n{'='*72}")
    print("Done.")


if __name__ == "__main__":
    main()
