#!/usr/bin/env python3
"""
Prepare the decoding-study input from the WMT25 test set.

Converts wmt25.jsonl (fields: src_text, prompt_instruction, src_lang, tgt_lang,
doc_id) into the mtoc input schema (doc_id, source_doc, tgt_lang, instruction)
and subsamples to a fixed number of segments per selected language pair.

Usage:
    python prepare_decoding_study.py \
        --input  /mnt/tg/data/projects/wmt26/llm-mt-dec/data/wmt25.jsonl \
        --output /mnt/tg/data/projects/wmt26/llm-mt-dec/data/decoding-study.jsonl \
        --lang-pairs en:ru_RU en:zh_CN \
        --per-pair 400 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--lang-pairs", nargs="+", default=["en:ru_RU", "en:zh_CN"],
                    help="src:tgt pairs to keep, e.g. en:ru_RU en:zh_CN")
    ap.add_argument("--per-pair", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    wanted = set(tuple(p.split(":", 1)) for p in args.lang_pairs)

    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(args.input) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("src_lang", ""), r.get("tgt_lang", ""))
            if key in wanted:
                by_pair[key].append(r)

    rng = random.Random(args.seed)
    out_rows: list[dict] = []
    for key in wanted:
        rows = by_pair.get(key, [])
        if not rows:
            print(f"WARNING: no rows for {key[0]}:{key[1]}")
            continue
        rng.shuffle(rows)
        sample = rows[: args.per_pair]
        for r in sample:
            out_rows.append({
                "doc_id": r["doc_id"],
                "source_doc": r["src_text"],
                "tgt_lang": r["tgt_lang"],
                "instruction": r.get("prompt_instruction", ""),
                # keep provenance for analysis
                "src_lang": r.get("src_lang", ""),
                "domain": r.get("domain", ""),
            })
        print(f"{key[0]}->{key[1]}: kept {len(sample)} of {len(rows)} available")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(out_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
