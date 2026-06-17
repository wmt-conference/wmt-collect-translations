#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

from io_utils import JsonlWriter, iter_jsonl, normalize_source_text, read_json
from interfaces import RuntimeModelConfig, TranslationRequest, TranslationResult, utc_timestamp
from offline import create_offline_backend


REQUIRED_INPUT_FIELDS = ("doc_id", "source_doc", "tgt_lang", "instruction")
DEFAULT_REGISTRY = Path(__file__).resolve().parent / "model_registry.wmt26.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"

@dataclass(frozen=True)
class Job:
    model_key: str
    config: RuntimeModelConfig
    output_path: Path
    failure_path: Path

def load_model_registry(path: Path) -> Dict[str, RuntimeModelConfig]:
    payload = read_json(path)
    models = payload.get("models")
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


def output_path_for(output_dir: Path, model_key: str) -> Path:
    return output_dir / model_key / f"{model_key}.translations.jsonl"


def failure_path_for(log_dir: Path, model_key: str) -> Path:
    return log_dir / model_key / f"{model_key}.failures.jsonl"


def load_completed_keys(output_path: Path, model_key: str) -> set[Tuple[str, str, str]]:
    if not output_path.exists():
        return set()
    completed: set[Tuple[str, str, str]] = set()
    for _, payload in iter_jsonl(output_path):
        doc_id = str(payload.get("doc_id", ""))
        tgt_lang = str(payload.get("tgt_lang", ""))
        row_model_key = str(payload.get("model_key", ""))
        if doc_id and tgt_lang and row_model_key == model_key:
            completed.add((doc_id, tgt_lang, row_model_key))
    return completed


def build_output_row(
    result: TranslationResult,
    *,
    backend: str,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, object]:
    record = result.request
    row = dict(record.raw)
    row.update({
        "doc_id": record.doc_id,
        "tgt_lang": record.tgt_lang,
        "model_key": result.model_key,
        "model": result.model_id,
        "backend": result.backend or backend,
        "translation": result.translation,
        "timestamp": utc_timestamp(),
        "generation_params": {
            "batch_size": batch_size,
            "max_input_length": max_input_length,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "metadata": result.metadata,
    })
    return row


def build_failure_row(record: TranslationRequest, model_config: RuntimeModelConfig, error: str) -> Dict[str, object]:
    return {
        "doc_id": record.doc_id,
        "tgt_lang": record.tgt_lang,
        "model_key": model_config.key,
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


def build_jobs(models: Sequence[RuntimeModelConfig], output_dir: Path, log_dir: Path) -> List[Job]:
    return [
        Job(
            model_key=model.key,
            config=model,
            output_path=output_path_for(output_dir, model.key),
            failure_path=failure_path_for(log_dir, model.key),
        )
        for model in models
    ]


def run_job(args: argparse.Namespace, job: Job, records: Sequence[TranslationRequest]) -> Tuple[int, int]:
    backend = args.backend if args.backend != "registry" else job.config.backend
    batch_size = args.batch_size if args.batch_size is not None else job.config.default_batch_size
    max_input_length = args.max_input_length if args.max_input_length is not None else job.config.default_max_input_length
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else job.config.default_max_new_tokens

    completed = load_completed_keys(job.output_path, job.model_key) if args.resume else set()
    pending = [
        record for record in records
        if (record.doc_id, record.tgt_lang, job.model_key) not in completed
    ]
    print(
        f"Job {job.model_key}: total={len(records)} pending={len(pending)} resumed={len(records) - len(pending)} output={job.output_path}",
        flush=True,
    )
    if not pending:
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

    success_count = 0
    failure_count = 0
    with JsonlWriter(job.output_path) as output_writer, JsonlWriter(job.failure_path) as failure_writer:
        for batch in iter_batches(pending, batch_size):
            try:
                results = backend_runner.translate_batch(
                    batch,
                    max_input_length=max_input_length,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                if len(results) != len(batch):
                    raise RuntimeError(f"expected {len(batch)} translations but received {len(results)}")
            except Exception as exc:
                for record in batch:
                    failure_writer.write(build_failure_row(record, job.config, f"batch failed: {exc}"))
                    failure_count += 1
                continue

            for result in results:
                if not result.translation:
                    failure_writer.write(build_failure_row(result.request, job.config, "empty translation"))
                    failure_count += 1
                    continue
                output_writer.write(build_output_row(
                    result,
                    backend=backend,
                    batch_size=batch_size,
                    max_input_length=max_input_length,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                ))
                success_count += 1

    print(f"Completed {job.model_key}: wrote={success_count} failures={failure_count}", flush=True)
    return success_count, failure_count


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
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
    ]
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
    if args.max_gpu_memory is not None:
        command.extend(["--max-gpu-memory", args.max_gpu_memory])
    if args.allow_cpu_offload:
        command.append("--allow-cpu-offload")
    if not args.resume:
        command.append("--no-resume")
    return command


def command_list_models(args: argparse.Namespace) -> int:
    registry = load_model_registry(args.model_registry)
    print("model_key\thf_id\tbackend\ttp\tbatch\tmax_input\tmax_new\ttrack\trun_scope\tgated\tnotes")
    for model in registry.values():
        print(
            f"{model.key}\t{model.model_id}\t{model.backend}\t{model.tensor_parallel_size}\t"
            f"{model.default_batch_size}\t{model.default_max_input_length}\t{model.default_max_new_tokens}\t"
            f"{model.track}\t{model.run_scope}\t"
            f"{str(model.gated).lower()}\t{model.notes}"
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

    jobs = build_jobs(selected_models, args.output_dir, args.log_dir)
    total_failures = 0
    total_written = 0
    for job in jobs:
        written, failures = run_job(args, job, records)
        total_written += written
        total_failures += failures

    print(f"All jobs finished: wrote={total_written} failures={total_failures}")
    return 2 if total_failures else 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with doc_id, source_doc, tgt_lang, and instruction fields.")
    parser.add_argument("--model-registry", type=Path, default=DEFAULT_REGISTRY, help="Path to model registry JSON file.")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model keys from the registry, or 'all'.")


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    add_common_args(parser)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root output directory.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Root failure-log directory.")
    parser.add_argument("--backend", choices=("registry", "hf", "vllm"), default="registry", help="Use registry backend or force hf/vllm.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of non-empty rows to run.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-model batch size.")
    parser.add_argument("--max-input-length", type=int, default=None, help="Override per-model prompt truncation length.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override per-model generation length.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature. 0 means greedy decoding.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling value when temperature > 0.")
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
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
