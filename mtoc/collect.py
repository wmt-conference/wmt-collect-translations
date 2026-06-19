#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import yaml

from io_utils import JsonlWriter, iter_jsonl, normalize_source_text, read_json
from interfaces import DecodingConfig, RuntimeModelConfig, TranslationRequest, TranslationResult, utc_timestamp
from offline import create_offline_backend


REQUIRED_INPUT_FIELDS = ("doc_id", "source_doc", "tgt_lang", "instruction")
DEFAULT_REGISTRY = Path(__file__).resolve().parent / "model_registry.wmt26.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"

@dataclass(frozen=True)
class Job:
    model_key: str
    config: RuntimeModelConfig
    variants: Tuple[DecodingConfig, ...]
    output_dir: Path
    log_dir: Path

def read_registry_file(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(text)
    return read_json(path)


VALID_DECODING_METHODS = ("greedy", "sample", "beam")


def parse_decoding_config(name: str, raw: object) -> DecodingConfig:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"variant '{name}' must be a mapping, got {type(raw).__name__}")
    method = str(raw.get("method", "greedy")).lower()
    if method not in VALID_DECODING_METHODS:
        raise ValueError(f"variant '{name}': method must be one of {VALID_DECODING_METHODS}, got '{method}'")
    thinking = raw.get("thinking", None)
    if thinking is not None:
        thinking = bool(thinking)
    return DecodingConfig(
        name=name,
        method=method,
        temperature=float(raw.get("temperature", 0.0)),
        top_p=float(raw.get("top_p", 1.0)),
        num_beams=int(raw.get("num_beams", 1)),
        thinking=thinking,
        seed=(int(raw["seed"]) if raw.get("seed") is not None else None),
        max_new_tokens=(int(raw["max_new_tokens"]) if raw.get("max_new_tokens") is not None else None),
    )


def parse_variants(raw_config: Dict[str, object]) -> Tuple[DecodingConfig, ...]:
    raw_variants = raw_config.get("variants")
    if not raw_variants:
        # No variants declared: a single implicit greedy variant that maps to the
        # bare model key (preserves the pre-variant output layout).
        return (DecodingConfig(name="default", method="greedy"),)
    if not isinstance(raw_variants, dict):
        raise ValueError("'variants' must be a mapping of variant_name -> decoding config")
    return tuple(parse_decoding_config(str(name), cfg) for name, cfg in raw_variants.items())


def load_model_registry(path: Path) -> Dict[str, RuntimeModelConfig]:
    payload = read_registry_file(path)
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, dict):
        raise ValueError(f"Model registry must contain a top-level 'models' object: {path}")

    registry: Dict[str, RuntimeModelConfig] = {}
    for key, raw_config in models.items():
        if not isinstance(raw_config, dict):
            raise ValueError(f"Model registry entry must be an object: {key}")
        registry[str(key)] = RuntimeModelConfig(
            key=str(key),
            model_id=str(raw_config["hf_id"]),
            backend=str(raw_config.get("backend", "hf")),
            dtype=str(raw_config.get("dtype", "bfloat16")),
            tensor_parallel_size=int(raw_config.get("tensor_parallel_size", 1)),
            default_batch_size=int(raw_config.get("default_batch_size", 1)),
            default_max_input_length=int(raw_config.get("default_max_input_length", 4096)),
            default_max_new_tokens=int(raw_config.get("default_max_new_tokens", 1024)),
            trust_remote_code=bool(raw_config.get("trust_remote_code", False)),
            gated=bool(raw_config.get("gated", False)),
            track=str(raw_config.get("track", "")),
            run_scope=str(raw_config.get("run_scope", "")),
            notes=str(raw_config.get("notes", "")),
            variants=parse_variants(raw_config),
        )
    return registry



def resolve_models(raw_models: Sequence[str], registry: Dict[str, RuntimeModelConfig]) -> List[RuntimeModelConfig]:
    if not raw_models or raw_models == ["all"]:
        return list(registry.values())

    resolved: List[RuntimeModelConfig] = []
    seen = set()
    for raw_model in raw_models:
        if raw_model == "all":
            return list(registry.values())
        if raw_model not in registry:
            available = ", ".join(sorted(registry))
            raise ValueError(f"Unknown model key '{raw_model}'. Available models: {available}")
        if raw_model not in seen:
            resolved.append(registry[raw_model])
            seen.add(raw_model)
    return resolved


def validate_input(path: Path) -> Dict[str, object]:
    issues: List[str] = []
    notes: List[str] = []
    row_count = 0
    multimodal_count = 0
    sample_keys: Tuple[str, ...] = ()

    for line_number, payload in iter_jsonl(path):
        row_count += 1
        if not sample_keys:
            sample_keys = tuple(payload.keys())
        missing = [field for field in REQUIRED_INPUT_FIELDS if field not in payload]
        if missing:
            doc_id = payload.get("doc_id", f"line {line_number}")
            issues.append(f"row {line_number} ({doc_id}) missing fields: {', '.join(missing)}")
        if payload.get("multimodal_instruction") or payload.get("multimodal_input_path"):
            multimodal_count += 1

    if row_count == 0:
        issues.append("input contains no JSONL records")
    if multimodal_count:
        notes.append(
            f"{multimodal_count} row(s) carry multimodal assets; translating the text "
            f"transcript only (audio/screenshot is ignored)."
        )

    return {"row_count": row_count, "sample_keys": sample_keys, "issues": issues, "notes": notes}


def load_input_records(path: Path, limit: int | None) -> List[TranslationRequest]:
    records: List[TranslationRequest] = []
    for line_number, payload in iter_jsonl(path):
        missing = [field for field in REQUIRED_INPUT_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields: {', '.join(missing)}")

        # Multimodal samples (spoken/social domains) carry optional audio or
        # screenshot assets, but always include a text transcript in source_doc.
        # We translate the transcript and ignore the asset (audio/screenshot is
        # not required per the task), so these rows are not dropped.
        source_doc = normalize_source_text(str(payload["source_doc"]))
        if not source_doc:
            continue
        records.append(TranslationRequest(
            doc_id=str(payload["doc_id"]),
            source_doc=source_doc,
            tgt_lang=str(payload["tgt_lang"]),
            instruction=str(payload["instruction"]),
            raw=dict(payload),
        ))
        if limit is not None and len(records) >= limit:
            break
    return records


def run_key_for(model_key: str, variant: DecodingConfig) -> str:
    """Output stream key for a (model, variant) pair.

    The implicit greedy "default" variant maps to the bare model key so a plain
    `run --models X` keeps the pre-variant output layout; named variants get a
    `<model>__<variant>` suffix so each is a self-contained, comparable stream.
    """
    if variant.name in ("", "default"):
        return model_key
    return f"{model_key}__{variant.name}"


def select_variants(model: RuntimeModelConfig, requested: Sequence[str] | None) -> Tuple[DecodingConfig, ...]:
    if not requested or list(requested) == ["all"]:
        return model.variants
    by_name = {v.name: v for v in model.variants}
    chosen: List[DecodingConfig] = []
    for name in requested:
        if name not in by_name:
            available = ", ".join(by_name) or "(none)"
            raise ValueError(f"{model.key}: unknown variant '{name}'. Available: {available}")
        chosen.append(by_name[name])
    return tuple(chosen)


def output_path_for(output_dir: Path, run_key: str) -> Path:
    return output_dir / run_key / f"{run_key}.translations.jsonl"


def failure_path_for(log_dir: Path, run_key: str) -> Path:
    return log_dir / run_key / f"{run_key}.failures.jsonl"


def load_completed_keys(output_path: Path, run_key: str) -> set[Tuple[str, str, str]]:
    if not output_path.exists():
        return set()
    completed: set[Tuple[str, str, str]] = set()
    for _, payload in iter_jsonl(output_path):
        doc_id = str(payload.get("doc_id", ""))
        tgt_lang = str(payload.get("tgt_lang", ""))
        # Match on the stream's run_key (falling back to model_key for files
        # written before variants existed).
        row_key = str(payload.get("run_key", payload.get("model_key", "")))
        if doc_id and tgt_lang and row_key == run_key:
            completed.add((doc_id, tgt_lang, row_key))
    return completed



def build_output_row(
    result: TranslationResult,
    *,
    run_key: str,
    backend: str,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    decoding: DecodingConfig,
) -> Dict[str, object]:
    record = result.request
    row = dict(record.raw)
    row.update({
        "doc_id": record.doc_id,
        "tgt_lang": record.tgt_lang,
        "model_key": result.model_key,
        "run_key": run_key,
        "variant": decoding.name,
        "model": result.model_id,
        "backend": result.backend or backend,
        "translation": result.translation,
        "timestamp": utc_timestamp(),
        "generation_params": {
            "batch_size": batch_size,
            "max_input_length": max_input_length,
            "max_new_tokens": max_new_tokens,
            "method": decoding.method,
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "num_beams": decoding.num_beams,
            "thinking": decoding.thinking,
            "seed": decoding.seed,
        },
        "metadata": result.metadata,
    })
    return row


def build_failure_row(record: TranslationRequest, model_config: RuntimeModelConfig, run_key: str, error: str) -> Dict[str, object]:
    return {
        "doc_id": record.doc_id,
        "tgt_lang": record.tgt_lang,
        "model_key": model_config.key,
        "run_key": run_key,
        "model": model_config.model_id,
        "error": error,
        "timestamp": utc_timestamp(),
        "row": record.raw,
    }


def iter_batches(records: Sequence[TranslationRequest], batch_size: int) -> Iterator[List[TranslationRequest]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(records), batch_size):
        yield list(records[start:start + batch_size])


def build_jobs(models: Sequence[RuntimeModelConfig], requested_variants: Sequence[str] | None,
               output_dir: Path, log_dir: Path) -> List[Job]:
    jobs: List[Job] = []
    for model in models:
        variants = select_variants(model, requested_variants)
        if not variants:
            continue
        jobs.append(Job(
            model_key=model.key,
            config=model,
            variants=variants,
            output_dir=output_dir,
            log_dir=log_dir,
        ))
    return jobs


def run_job(args: argparse.Namespace, job: Job, records: Sequence[TranslationRequest]) -> Tuple[int, int]:
    backend = args.backend if args.backend != "registry" else job.config.backend
    batch_size = args.batch_size if args.batch_size is not None else job.config.default_batch_size
    max_input_length = args.max_input_length if args.max_input_length is not None else job.config.default_max_input_length

    # Figure out which variants still have pending work, so a model is only loaded
    # when at least one of its variants needs to run.
    plan: List[Tuple[DecodingConfig, str, Path, Path, List[TranslationRequest], int]] = []
    for variant in job.variants:
        run_key = run_key_for(job.model_key, variant)
        output_path = output_path_for(job.output_dir, run_key)
        failure_path = failure_path_for(job.log_dir, run_key)
        max_new_tokens = (
            args.max_new_tokens if args.max_new_tokens is not None
            else (variant.max_new_tokens if variant.max_new_tokens is not None else job.config.default_max_new_tokens)
        )
        completed = load_completed_keys(output_path, run_key) if args.resume else set()
        pending = [r for r in records if (r.doc_id, r.tgt_lang, run_key) not in completed]
        print(
            f"Job {run_key}: total={len(records)} pending={len(pending)} "
            f"resumed={len(records) - len(pending)} method={variant.method} output={output_path}",
            flush=True,
        )
        if pending:
            plan.append((variant, run_key, output_path, failure_path, pending, max_new_tokens))

    if not plan:
        return 0, 0

    backend_runner = create_offline_backend(
        job.config,
        backend_override=args.backend,
        device_map=args.device_map,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_memory_per_gpu=args.max_gpu_memory,
        allow_cpu_offload=args.allow_cpu_offload,
    )
    backend_runner.load()

    total_success = 0
    total_failure = 0
    for variant, run_key, output_path, failure_path, pending, max_new_tokens in plan:
        success_count = 0
        failure_count = 0
        with JsonlWriter(output_path) as output_writer, JsonlWriter(failure_path) as failure_writer:
            for batch in iter_batches(pending, batch_size):
                try:
                    results = backend_runner.translate_batch(
                        batch,
                        max_input_length=max_input_length,
                        max_new_tokens=max_new_tokens,
                        decoding=variant,
                    )
                    if len(results) != len(batch):
                        raise RuntimeError(f"expected {len(batch)} translations but received {len(results)}")
                except Exception as exc:
                    for record in batch:
                        failure_writer.write(build_failure_row(record, job.config, run_key, f"batch failed: {exc}"))
                        failure_count += 1
                    continue

                for result in results:
                    if not result.translation:
                        failure_writer.write(build_failure_row(result.request, job.config, run_key, "empty translation"))
                        failure_count += 1
                        continue
                    output_writer.write(build_output_row(
                        result,
                        run_key=run_key,
                        backend=backend,
                        batch_size=batch_size,
                        max_input_length=max_input_length,
                        max_new_tokens=max_new_tokens,
                        decoding=variant,
                    ))
                    success_count += 1
        print(f"Completed {run_key}: wrote={success_count} failures={failure_count}", flush=True)
        total_success += success_count
        total_failure += failure_count
    return total_success, total_failure


def parse_gpu_ids(raw_gpus: str) -> List[str]:
    gpu_ids = [gpu.strip() for gpu in raw_gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpu_ids


def allocate_gpu_waves(models: Sequence[RuntimeModelConfig], gpu_ids: Sequence[str]) -> List[List[Tuple[RuntimeModelConfig, List[str]]]]:
    waves: List[List[Tuple[RuntimeModelConfig, List[str]]]] = []
    current_wave: List[Tuple[RuntimeModelConfig, List[str]]] = []
    cursor = 0
    for model in models:
        need = model.tensor_parallel_size
        if need > len(gpu_ids):
            raise ValueError(f"{model.key} requests {need} GPUs, but only {len(gpu_ids)} were provided")
        if cursor + need > len(gpu_ids):
            waves.append(current_wave)
            current_wave = []
            cursor = 0
        current_wave.append((model, list(gpu_ids[cursor:cursor + need])))
        cursor += need
    if current_wave:
        waves.append(current_wave)
    return waves


def quote_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_run_command(args: argparse.Namespace, model: RuntimeModelConfig) -> List[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--input",
        str(args.input.resolve()),
        "--model-registry",
        str(args.model_registry.resolve()),
        "--models",
        model.key,
        "--output-dir",
        str(args.output_dir.resolve()),
        "--log-dir",
        str(args.log_dir.resolve()),
        "--backend",
        args.backend,
    ]
    if args.variants and list(args.variants) != ["all"]:
        command.extend(["--variants", *args.variants])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.max_input_length is not None:
        command.extend(["--max-input-length", str(args.max_input_length)])
    if args.max_new_tokens is not None:
        command.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.device_map != "auto":
        command.extend(["--device-map", args.device_map])
    command.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.max_gpu_memory is not None:
        command.extend(["--max-gpu-memory", args.max_gpu_memory])
    if args.allow_cpu_offload:
        command.append("--allow-cpu-offload")
    if not args.resume:
        command.append("--no-resume")
    return command


def command_list_models(args: argparse.Namespace) -> int:
    registry = load_model_registry(args.model_registry)
    print("model_key\thf_id\tbackend\ttp\tbatch\tmax_input\tmax_new\ttrack\trun_scope\tgated\tvariants\tnotes")
    for model in registry.values():
        variants = ",".join(v.name for v in model.variants)
        print(
            f"{model.key}\t{model.model_id}\t{model.backend}\t{model.tensor_parallel_size}\t"
            f"{model.default_batch_size}\t{model.default_max_input_length}\t{model.default_max_new_tokens}\t"
            f"{model.track}\t{model.run_scope}\t"
            f"{str(model.gated).lower()}\t{variants}\t{model.notes}"
        )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    registry = load_model_registry(args.model_registry)
    selected_models = resolve_models(args.models, registry)
    input_result = validate_input(args.input)

    print(f"Input: {args.input}")
    print(f"Rows: {input_result['row_count']}")
    print("Input sample keys: " + (", ".join(input_result["sample_keys"]) if input_result["sample_keys"] else "n/a"))
    print("Selected models: " + ", ".join(model.key for model in selected_models))
    print("Required GPUs if run sequentially by model: " + ", ".join(f"{model.key}:{model.tensor_parallel_size}" for model in selected_models))

    issues = list(input_result["issues"])
    notes = list(input_result.get("notes", []))
    if notes:
        print("Notes:")
        for note in notes:
            print(f"- {note}")
    if issues:
        print("Errors:")
        for issue in issues:
            print(f"- {issue}")
        return 2
    print("Validation passed.")
    return 0


def command_plan(args: argparse.Namespace) -> int:
    registry = load_model_registry(args.model_registry)
    selected_models = resolve_models(args.models, registry)
    gpu_ids = parse_gpu_ids(args.gpus)
    waves = allocate_gpu_waves(selected_models, gpu_ids)

    print(f"Input: {args.input}")
    print(f"GPU pool: {','.join(gpu_ids)}")
    print("Planned waves:")
    for wave_index, wave in enumerate(waves, start=1):
        print(f"\n# Wave {wave_index}: {len(wave)} job(s), run these concurrently")
        for model, model_gpus in wave:
            command = build_run_command(args, model)
            env = f"CUDA_VISIBLE_DEVICES={','.join(model_gpus)}"
            print(f"{env} {quote_command(command)}")

    if args.gnu_parallel:
        print("\nGNU parallel form:")
        print("# Write each wave to its own commands.waveN.txt file and run the waves sequentially.")
        for wave_index, wave in enumerate(waves, start=1):
            print(f"# parallel --jobs {len(wave)} --halt soon,fail=1 < commands.wave{wave_index}.txt")
    return 0


def command_run(args: argparse.Namespace) -> int:
    registry = load_model_registry(args.model_registry)
    selected_models = resolve_models(args.models, registry)
    records = load_input_records(args.input, args.limit)
    if not records:
        print("No non-empty source_doc records selected.", file=sys.stderr)
        return 2

    jobs = build_jobs(selected_models, args.variants, args.output_dir, args.log_dir)
    total_failures = 0
    total_written = 0
    for job in jobs:
        written, failures = run_job(args, job, records)
        total_written += written
        total_failures += failures

    print(f"All jobs finished: wrote={total_written} failures={total_failures}")
    return 2 if total_failures else 0


def model_weights_cached(model: RuntimeModelConfig) -> bool:
    """Return True if the model's weights appear to be present in the local HF cache.

    Avoids launching a subprocess for a model whose checkpoint was never downloaded
    (e.g. a gated repo we lack access to). We look for a non-empty snapshot dir with
    at least one weight shard under the configured HF cache.
    """
    cache_root = os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE")
    if not cache_root:
        # No offline cache configured; assume the runtime will fetch as needed.
        return True
    snapshots = Path(cache_root) / ("models--" + model.model_id.replace("/", "--")) / "snapshots"
    if not snapshots.is_dir():
        return False
    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir():
            continue
        for pattern in ("*.safetensors", "*.bin", "*.gguf", "consolidated*.pth"):
            if any(snapshot.glob(pattern)):
                return True
    return False


def command_launch(args: argparse.Namespace) -> int:
    """Run each selected model in its own subprocess.

    Running every model inside a single process leaks GPU memory between models
    (vLLM workers in particular stay resident), so each model is isolated in a
    fresh process. Single-GPU (tp=1) models run concurrently across the GPU pool
    (one model per GPU, refilling as GPUs free up); multi-GPU models then run
    sequentially, each reserving the GPUs it needs. Resume is enabled, so
    re-launching skips rows already written.

    Models whose weights are not present in the local cache are skipped (so an
    undownloaded gated repo like tiny_aya_global is ignored automatically) unless
    --require-cached is disabled.
    """
    registry = load_model_registry(args.model_registry)
    selected_models = resolve_models(args.models, registry)
    gpu_ids = parse_gpu_ids(args.gpus)

    runnable: List[RuntimeModelConfig] = []
    skipped: List[str] = []
    for model in selected_models:
        if model.tensor_parallel_size > len(gpu_ids):
            skipped.append(f"{model.key} (needs {model.tensor_parallel_size} GPUs, have {len(gpu_ids)})")
            continue
        if args.require_cached and not model_weights_cached(model):
            skipped.append(f"{model.key} (weights not in local cache)")
            continue
        runnable.append(model)

    print(f"Launcher: {len(runnable)} model(s) to run, {len(skipped)} skipped")
    for note in skipped:
        print(f"  skip {note}")
    if not runnable:
        print("No runnable models selected.", file=sys.stderr)
        return 2

    # Run small models first so they pack the pool while big models wait for room.
    runnable.sort(key=lambda m: m.tensor_parallel_size)
    print(
        f"Scheduling across {len(gpu_ids)} GPU(s): models run concurrently whenever "
        f"enough GPUs are free (tp sizes: "
        f"{', '.join(f'{m.key}={m.tensor_parallel_size}' for m in runnable)}).",
        flush=True,
    )

    results = _run_pool(args, runnable, gpu_ids)

    print("\nLauncher summary:")
    failures = 0
    for key, code in results:
        print(f"  {key:20} {'ok' if code == 0 else f'FAILED ({code})'}")
        if code != 0:
            failures += 1
    print(f"Completed {len(results) - failures}/{len(results)} models successfully.")
    return 2 if failures else 0


def _run_pool(
    args: argparse.Namespace, models: Sequence[RuntimeModelConfig], gpu_ids: Sequence[str]
) -> List[Tuple[str, int]]:
    """Dynamic GPU-pool scheduler for models of any tensor_parallel_size.

    A model launches as soon as its tp GPUs are free, so multiple models run
    concurrently whenever the pool allows (e.g. two tp=2 models share 4 GPUs),
    while a tp=8 model waits until the whole node is free. Each model's console
    output goes to <log-dir>/<key>/run.log so parallel streams don't interleave.
    To keep larger models from starving behind a trickle of smaller ones, models
    are tried in tp order and a model only yields to those ahead of it in the
    queue (head-of-line), not to every later model.
    """
    free_gpus: List[str] = list(gpu_ids)
    pending: List[RuntimeModelConfig] = list(models)
    running: List[Tuple[RuntimeModelConfig, List[str], Any, Any]] = []  # model, gpus, proc, logfile
    results: List[Tuple[str, int]] = []
    done = 0
    total = len(models)

    while pending or running:
        # Launch the head of the queue (and any later model that fits) without
        # letting a smaller model jump ahead of a blocked larger one.
        index = 0
        while index < len(pending):
            model = pending[index]
            need = model.tensor_parallel_size
            if need <= len(free_gpus):
                pending.pop(index)
                gpus = [free_gpus.pop(0) for _ in range(need)]
                command = build_run_command(args, model)
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
                log_path = failure_path_for(args.log_dir, model.key).parent / "run.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                logfile = open(log_path, "w", encoding="utf-8")
                proc = subprocess.Popen(command, env=env, stdout=logfile, stderr=subprocess.STDOUT)
                print(
                    f">> {model.key} started on GPU(s) {','.join(gpus)} "
                    f"({model.backend} backend); log {log_path}",
                    flush=True,
                )
                running.append((model, gpus, proc, logfile))
            else:
                # Head-of-line: stop so this model isn't starved by later ones.
                break

        # Reap finished processes and return their GPUs to the pool.
        still_running: List[Tuple[RuntimeModelConfig, List[str], Any, Any]] = []
        for model, gpus, proc, logfile in running:
            code = proc.poll()
            if code is None:
                still_running.append((model, gpus, proc, logfile))
                continue
            logfile.close()
            free_gpus.extend(gpus)
            done += 1
            status = "ok" if code == 0 else f"FAILED (exit {code})"
            print(f"<< [{done}/{total}] {model.key}: {status} (GPU(s) {','.join(gpus)} freed)", flush=True)
            results.append((model.key, code))
        running = still_running

        if pending or running:
            # Settle delay also lets the driver reclaim freed GPU memory before the
            # next (possibly vLLM) launch checks free memory.
            time.sleep(5)

    return results




def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with doc_id, source_doc, tgt_lang, and instruction fields.")
    parser.add_argument("--model-registry", type=Path, default=DEFAULT_REGISTRY, help="Path to model registry JSON file.")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model keys from the registry, or 'all'.")


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    add_common_args(parser)
    parser.add_argument("--variants", nargs="+", default=["all"], help="Decoding variant names from the registry to run, or 'all'. Each variant is a separate output stream.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root output directory.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Root failure-log directory.")
    parser.add_argument("--backend", choices=("registry", "auto", "hf", "vllm"), default="registry", help="Use the registry backend, or force auto (vLLM with HF fallback), hf, or vllm.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of non-empty rows to run.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-model batch size.")
    parser.add_argument("--max-input-length", type=int, default=None, help="Override per-model prompt truncation length.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override per-model/variant generation length.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map for --backend hf (e.g. auto, balanced).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Fraction of each visible GPU's memory Accelerate/vLLM may use.")
    parser.add_argument("--max-gpu-memory", default=None, help="Per-GPU memory cap for --backend hf (e.g. '72GiB'). Overrides --gpu-memory-utilization when set.")
    parser.add_argument("--allow-cpu-offload", action="store_true", help="Permit Accelerate to offload layers to CPU/disk if a model does not fit on the visible GPUs (slow).")
    parser.set_defaults(resume=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not skip rows already present in output JSONL.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect translations from local/open-source LLMs with Transformers or vLLM.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-models", help="List registered models and runtime defaults.")
    list_parser.add_argument("--model-registry", type=Path, default=DEFAULT_REGISTRY, help="Path to model registry JSON file.")

    validate_parser = subparsers.add_parser("validate", help="Validate input schema and selected models.")
    add_common_args(validate_parser)

    plan_parser = subparsers.add_parser("plan", help="Print one runnable command per selected model with GPU assignments.")
    add_runtime_args(plan_parser)
    plan_parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"), help="Comma-separated GPU ids available for planning.")
    plan_parser.add_argument("--jobs", type=int, default=8, help="Suggested GNU parallel job count for printed commands.")
    plan_parser.add_argument("--gnu-parallel", action="store_true", help="Also print GNU parallel hint.")

    run_parser = subparsers.add_parser("run", help="Run selected models sequentially in this process.")
    add_runtime_args(run_parser)

    launch_parser = subparsers.add_parser("launch", help="Run each selected model in its own subprocess (frees GPU memory between models). Safe for full-set runs.")
    add_runtime_args(launch_parser)
    launch_parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"), help="Comma-separated GPU ids available to the launcher.")
    launch_parser.set_defaults(require_cached=True)
    launch_parser.add_argument("--no-require-cached", dest="require_cached", action="store_false", help="Also launch models whose weights are not in the local cache (the runtime will try to fetch them).")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-models":
        return command_list_models(args)
    if args.command == "validate":
        return command_validate(args)
    if args.command == "plan":
        return command_plan(args)
    if args.command == "run":
        return command_run(args)
    if args.command == "launch":
        return command_launch(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
