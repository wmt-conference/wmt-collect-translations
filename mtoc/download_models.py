#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_REGISTRY = Path(__file__).resolve().parent / "model_registry.wmt26.json"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "models" / "hf-hub"


def load_registry(path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"Registry must contain a top-level models object: {path}")
    return {str(key): value for key, value in models.items() if isinstance(value, dict)}


def resolve_models(raw_models: Sequence[str], registry: Dict[str, Dict[str, object]]) -> List[str]:
    if not raw_models or raw_models == ["all"]:
        return list(registry.keys())

    resolved: List[str] = []
    seen = set()
    for model_key in raw_models:
        if model_key == "all":
            return list(registry.keys())
        if model_key not in registry:
            available = ", ".join(sorted(registry))
            raise ValueError(f"Unknown model key '{model_key}'. Available models: {available}")
        if model_key not in seen:
            resolved.append(model_key)
            seen.add(model_key)
    return resolved


def configure_fast_hf(cache_dir: Path) -> None:
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def download_model(
    *,
    model_key: str,
    hf_id: str,
    cache_dir: Path,
    revision: str | None,
    max_workers: int,
    force_download: bool,
    local_files_only: bool,
) -> tuple[str, str, str]:
    from huggingface_hub import snapshot_download

    try:
        snapshot_path = snapshot_download(
            repo_id=hf_id,
            repo_type="model",
            revision=revision,
            cache_dir=str(cache_dir),
            max_workers=max_workers,
            force_download=force_download,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        return model_key, hf_id, f"FAILED\t{exc}"
    return model_key, hf_id, f"OK\t{snapshot_path}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download mtoc WMT26 Hugging Face models into a local HF cache.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="Model registry JSON file.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="HF cache directory to populate.")
    parser.add_argument("--models", nargs="+", default=["all"], help="Registry model keys to download, or 'all'.")
    parser.add_argument("--revision", default=None, help="Optional HF revision to download for every model.")
    parser.add_argument("--max-workers", type=int, default=8, help="Per-model concurrent file download workers.")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if cached.")
    parser.add_argument("--local-files-only", action="store_true", help="Check only existing local cache without network access.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected models and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = load_registry(args.registry)
    model_keys = resolve_models(args.models, registry)

    cache_dir = args.cache_dir.resolve()
    configure_fast_hf(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Registry: {args.registry}")
    print(f"HF cache: {cache_dir}")
    print(f"Models: {', '.join(model_keys)}")
    print(f"HF_HOME={os.environ.get('HF_HOME', '')}")
    print(f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
    print(f"HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
    print(f"HF_XET_HIGH_PERFORMANCE={os.environ.get('HF_XET_HIGH_PERFORMANCE')}")

    if args.dry_run:
        for model_key in model_keys:
            print(f"{model_key}\t{registry[model_key]['hf_id']}")
        return 0

    failures = 0
    for index, model_key in enumerate(model_keys, start=1):
        hf_id = str(registry[model_key]["hf_id"])
        print(f"\n[{index}/{len(model_keys)}] {model_key}: {hf_id}", flush=True)
        model_key, hf_id, status = download_model(
            model_key=model_key,
            hf_id=hf_id,
            cache_dir=cache_dir,
            revision=args.revision,
            max_workers=args.max_workers,
            force_download=args.force_download,
            local_files_only=args.local_files_only,
        )
        print(f"{model_key}\t{hf_id}\t{status}", flush=True)
        if status.startswith("FAILED"):
            failures += 1

    if failures:
        print(f"\nCompleted with {failures} failed download(s).", file=sys.stderr)
        return 2
    print("\nAll selected models are available in the HF cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())