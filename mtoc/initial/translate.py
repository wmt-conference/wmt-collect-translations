#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Iterator, List, Sequence

from io_utils import (
    JsonlWriter,
    iter_segments,
    load_completed_keys,
    normalize_source_text,
    read_json,
    validate_input_schema,
)
from translation_config import (
    DEFAULT_MIN_SHARED_MODEL_SUPPORT,
    MODEL_CONFIGS,
    REQUIRED_INPUT_FIELDS,
    VALID_MODEL_STATUS,
    default_log_dir,
    default_output_dir,
    default_registry_path,
)


BaseAdapter = Any


@dataclass(frozen=True)
class TranslationWriteEvent:
    destination: str
    payload: Dict[str, object]


class TranslationRecordWriter:
    def __init__(self, output_path: Path, failure_path: Path) -> None:
        self.output_writer = JsonlWriter(output_path)
        self.failure_writer = JsonlWriter(failure_path)

    def __enter__(self) -> "TranslationRecordWriter":
        self.output_writer.__enter__()
        self.failure_writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.output_writer.__exit__(exc_type, exc_value, traceback)
        self.failure_writer.__exit__(exc_type, exc_value, traceback)

    def consume(self, events: Iterable[TranslationWriteEvent]) -> tuple[int, int]:
        success_count = 0
        failure_count = 0
        for event in events:
            if event.destination == "output":
                self.output_writer.write(event.payload)
                success_count += 1
                continue
            if event.destination == "failure":
                self.failure_writer.write(event.payload)
                failure_count += 1
                continue
            raise ValueError(f"Unknown write destination: {event.destination}")
        return success_count, failure_count


def load_model_adapters_module() -> ModuleType:
    module_name = "wmt26_local_llm_translation_model_adapters"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    module_path = Path(__file__).resolve().parent / "model_adapters.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model adapter module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def create_adapter(model_key: str, device_map: str) -> BaseAdapter:
    module = load_model_adapters_module()
    return module.create_adapter(model_key, device_map=device_map)


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=default_registry_path(),
        help="Path to the language registry JSON file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model keys to include. Use 'all' or one or more of: nllb tower aya.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["all"],
        help="Target languages to include. Use 'all', task names, BCP47 tags, or display names.",
    )
    parser.add_argument(
        "--min-supported-models",
        type=int,
        default=DEFAULT_MIN_SHARED_MODEL_SUPPORT,
        help="Minimum number of selected models that must support a target for it to remain runnable. Default: 2.",
    )


def add_execution_args(
    parser: argparse.ArgumentParser,
    *,
    default_limit: int | None,
    default_priority: str,
) -> None:
    add_selection_args(parser)
    parser.add_argument(
        "--priority",
        choices=("all", "1", "2"),
        default=default_priority,
        help="Which priority bucket to execute. 'all' runs priority 1 and 2 targets.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=default_limit,
        help="Maximum number of non-empty segments to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Root directory for translation outputs.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=default_log_dir(),
        help="Root directory for failure logs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-model default batch size.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=None,
        help="Override per-model default max input length.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override per-model default max generated tokens.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed through to transformers model loading.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the runnable jobs without loading models or generating outputs.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resumable skip logic and rewrite the selected jobs from scratch.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and run the WMT26 translation pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate the language registry, selected targets, selected models, and input file schema.",
    )
    add_selection_args(validate_parser)

    list_parser = subparsers.add_parser(
        "list-targets",
        help="Print the canonical target names known by the registry.",
    )
    list_parser.add_argument(
        "--registry",
        type=Path,
        default=default_registry_path(),
        help="Path to the language registry JSON file.",
    )

    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Run a small end-to-end translation smoke test using the shared-language policy.",
    )
    add_execution_args(smoke_parser, default_limit=3, default_priority="1")

    run_parser = subparsers.add_parser(
        "run",
        help="Run translation for the runnable targets produced by the shared-language policy.",
    )
    add_execution_args(run_parser, default_limit=None, default_priority="all")

    return parser


def normalize_token(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_registry(registry_path: Path) -> Dict[str, object]:
    payload = read_json(registry_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Registry must be a JSON object: {registry_path}")
    return payload


def validate_registry_schema(registry: Dict[str, object], registry_path: Path) -> List[str]:
    issues: List[str] = []
    languages = registry.get("languages")
    source_language = registry.get("source_language")

    if not isinstance(source_language, dict):
        issues.append(f"{registry_path}: top-level source_language object is missing")

    if not isinstance(languages, dict):
        issues.append(f"{registry_path}: top-level languages object is missing")
        return issues

    required_entry_fields = (
        "task_name",
        "display_name",
        "bcp47",
        "aliases",
        "source_lang",
        "nllb_code",
        "tower_label",
        "aya_label",
        "supported_by",
        "model_status",
        "notes",
    )

    for task_name, entry in languages.items():
        if not isinstance(entry, dict):
            issues.append(f"{registry_path}: language entry {task_name} must be an object")
            continue

        missing_fields = [field for field in required_entry_fields if field not in entry]
        if missing_fields:
            issues.append(
                f"{registry_path}: language entry {task_name} is missing fields: {', '.join(missing_fields)}"
            )
            continue

        if entry["task_name"] != task_name:
            issues.append(
                f"{registry_path}: language entry key {task_name} does not match task_name {entry['task_name']}"
            )

        model_status = entry["model_status"]
        if not isinstance(model_status, dict):
            issues.append(f"{registry_path}: language entry {task_name} has invalid model_status")
            continue

        for model_key in MODEL_CONFIGS:
            if model_key not in model_status:
                issues.append(
                    f"{registry_path}: language entry {task_name} is missing model_status for {model_key}"
                )
                continue

            status = model_status[model_key]
            if status not in VALID_MODEL_STATUS:
                issues.append(
                    f"{registry_path}: language entry {task_name} has invalid status '{status}' for {model_key}"
                )

        supported_by = entry["supported_by"]
        if not isinstance(supported_by, list):
            issues.append(f"{registry_path}: language entry {task_name} supported_by must be a list")
            continue

        verified_models = sorted(
            model_key for model_key, status in model_status.items() if status == "verified"
        )
        if sorted(supported_by) != verified_models:
            issues.append(
                f"{registry_path}: language entry {task_name} supported_by must match verified model_status values"
            )

        if model_status.get("nllb") == "verified" and not entry.get("nllb_code"):
            issues.append(
                f"{registry_path}: language entry {task_name} needs nllb_code when NLLB status is verified"
            )

        if model_status.get("tower") != "blocked" and not entry.get("tower_label"):
            issues.append(
                f"{registry_path}: language entry {task_name} needs tower_label when Tower status is not blocked"
            )

        if model_status.get("aya") != "blocked" and not entry.get("aya_label"):
            issues.append(
                f"{registry_path}: language entry {task_name} needs aya_label when Aya status is not blocked"
            )

    return issues


def build_target_index(languages: Dict[str, Dict[str, object]]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for task_name, entry in languages.items():
        candidates = [task_name, str(entry["display_name"]), str(entry["bcp47"])]
        aliases = entry.get("aliases", [])
        if isinstance(aliases, list):
            candidates.extend(str(alias) for alias in aliases)

        for candidate in candidates:
            index[normalize_token(candidate)] = task_name
    return index


def resolve_models(raw_models: Sequence[str]) -> List[str]:
    tokens = [token.strip().lower() for token in raw_models]
    if not tokens or tokens == ["all"]:
        return list(MODEL_CONFIGS.keys())

    resolved: List[str] = []
    seen = set()
    for token in tokens:
        if token == "all":
            return list(MODEL_CONFIGS.keys())
        if token not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model key '{token}'. Expected one or more of: {', '.join(MODEL_CONFIGS)}"
            )
        if token not in seen:
            resolved.append(token)
            seen.add(token)
    return resolved


def resolve_targets(
    raw_targets: Sequence[str],
    languages: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    if not raw_targets or raw_targets == ["all"]:
        return list(languages.values())

    index = build_target_index(languages)
    resolved: List[Dict[str, object]] = []
    seen = set()

    for raw_target in raw_targets:
        if raw_target.lower() == "all":
            return list(languages.values())

        token = normalize_token(raw_target)
        if token not in index:
            available = ", ".join(languages.keys())
            raise ValueError(
                f"Unknown target '{raw_target}'. Use 'all' or one of the registry task names: {available}"
            )

        task_name = index[token]
        if task_name not in seen:
            resolved.append(languages[task_name])
            seen.add(task_name)
    return resolved


def supported_models_for_target(target: Dict[str, object], models: Sequence[str]) -> List[str]:
    supported_by = target.get("supported_by", [])
    if not isinstance(supported_by, list):
        return []
    return [model_key for model_key in models if model_key in supported_by]


def build_execution_plan(
    targets: Iterable[Dict[str, object]],
    models: Sequence[str],
    min_supported_models: int,
) -> Dict[str, object]:
    priority_1: List[Dict[str, object]] = []
    priority_2: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    threshold = min(max(min_supported_models, 1), len(models))

    for target in targets:
        eligible_models = supported_models_for_target(target, models)
        entry = dict(target)
        entry["eligible_models"] = eligible_models

        if len(eligible_models) == len(models):
            priority_1.append(entry)
        elif len(eligible_models) >= threshold:
            priority_2.append(entry)
        else:
            skipped.append(entry)

    return {
        "priority_1": priority_1,
        "priority_2": priority_2,
        "skipped": skipped,
        "threshold": threshold,
    }


def format_plan_entries(entries: Sequence[Dict[str, object]]) -> str:
    if not entries:
        return "none"

    parts: List[str] = []
    for entry in entries:
        models = ",".join(str(model) for model in entry.get("eligible_models", []))
        task_name = str(entry.get("task_name", "unknown"))
        parts.append(f"{task_name}[{models}]" if models else task_name)
    return ", ".join(parts)


def skipped_messages(entries: Sequence[Dict[str, object]]) -> List[str]:
    messages: List[str] = []
    for entry in entries:
        notes = entry.get("notes", [])
        note_text = "; ".join(str(note) for note in notes) if isinstance(notes, list) and notes else "No note recorded."
        messages.append(
            f"{entry['task_name']} ({entry['display_name']}): below shared-language threshold. {note_text}"
        )
    return messages


def collect_selection_context(args: argparse.Namespace) -> Dict[str, object]:
    registry = load_registry(args.registry)
    registry_issues = validate_registry_schema(registry, args.registry)

    input_issues: List[str] = []
    input_summary = {"row_count": 0, "sample_keys": ()}
    if not args.input.exists():
        input_issues.append(f"Input file does not exist: {args.input}")
    else:
        schema_result = validate_input_schema(args.input, REQUIRED_INPUT_FIELDS)
        input_summary = {
            "row_count": schema_result["row_count"],
            "sample_keys": schema_result["sample_keys"],
        }
        input_issues.extend(schema_result["issues"])

    languages = registry.get("languages", {})
    if not isinstance(languages, dict):
        languages = {}

    selection_errors: List[str] = []
    selected_models: List[str] = []
    selected_targets: List[Dict[str, object]] = []
    execution_plan: Dict[str, object] = {
        "priority_1": [],
        "priority_2": [],
        "skipped": [],
        "threshold": args.min_supported_models,
    }

    if not registry_issues:
        try:
            selected_models = resolve_models(args.models)
        except ValueError as exc:
            selection_errors.append(str(exc))

        try:
            selected_targets = resolve_targets(args.targets, languages)
        except ValueError as exc:
            selection_errors.append(str(exc))

    if not registry_issues and not selection_errors:
        execution_plan = build_execution_plan(
            targets=selected_targets,
            models=selected_models,
            min_supported_models=args.min_supported_models,
        )

    runnable_target_count = len(execution_plan["priority_1"]) + len(execution_plan["priority_2"])
    policy_errors: List[str] = []
    if not registry_issues and not input_issues and not selection_errors and runnable_target_count == 0:
        policy_errors.append(
            "No targets remain after applying the shared-language policy. Lower --min-supported-models or expand model coverage."
        )

    return {
        "registry": registry,
        "input_summary": input_summary,
        "selected_models": selected_models,
        "selected_targets": selected_targets,
        "execution_plan": execution_plan,
        "errors": registry_issues + input_issues + selection_errors + policy_errors,
    }


def print_selection_summary(context: Dict[str, object], args: argparse.Namespace) -> None:
    selected_targets = context["selected_targets"]
    execution_plan = context["execution_plan"]
    runnable_target_count = len(execution_plan["priority_1"]) + len(execution_plan["priority_2"])

    print(f"Registry: {args.registry}")
    print(f"Input: {args.input}")
    print(
        "Selected models: "
        + (
            ", ".join(str(model) for model in context["selected_models"])
            if context["selected_models"]
            else "n/a"
        )
    )
    print(
        "Requested targets: "
        + (
            ", ".join(str(target["task_name"]) for target in selected_targets)
            if selected_targets
            else "n/a"
        )
    )
    print(f"Input rows: {context['input_summary']['row_count']}")
    print(
        "Input sample keys: "
        + (
            ", ".join(str(key) for key in context["input_summary"]["sample_keys"])
            if context["input_summary"]["sample_keys"]
            else "n/a"
        )
    )
    print(f"Shared-language threshold: {execution_plan['threshold']}")
    print(f"Priority 1 targets: {format_plan_entries(execution_plan['priority_1'])}")
    print(f"Priority 2 targets: {format_plan_entries(execution_plan['priority_2'])}")
    print(f"Skipped targets: {format_plan_entries(execution_plan['skipped'])}")
    print(f"Runnable targets: {runnable_target_count}")

    skipped_details = skipped_messages(execution_plan["skipped"])
    if skipped_details:
        print("Skipped details:")
        for detail in skipped_details:
            print(f"- {detail}")


def selected_plan_entries(execution_plan: Dict[str, object], priority: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if priority in ("all", "1"):
        entries.extend(execution_plan["priority_1"])
    if priority in ("all", "2"):
        entries.extend(execution_plan["priority_2"])
    return entries


def iter_batches(items: Sequence[Dict[str, object]], batch_size: int) -> Iterator[List[Dict[str, object]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def load_source_segments(input_path: Path, limit: int | None) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    for payload in iter_segments(input_path):
        normalized_text = normalize_source_text(str(payload.get("seg", "")))
        if not normalized_text:
            continue
        entry = dict(payload)
        entry["seg"] = normalized_text
        segments.append(entry)
        if limit is not None and len(segments) >= limit:
            break
    return segments


def output_path_for(output_dir: Path, model_key: str, target_name: str) -> Path:
    return output_dir / model_key / f"{target_name}.jsonl"


def failure_path_for(log_dir: Path, model_key: str, target_name: str) -> Path:
    return log_dir / model_key / f"{target_name}.failures.jsonl"


def build_output_row(
    segment: Dict[str, object],
    target: Dict[str, object],
    source_language: Dict[str, object],
    model_key: str,
    adapter: BaseAdapter,
    translation: str,
    *,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    device_map: str,
) -> Dict[str, object]:
    row: Dict[str, object] = dict(segment)
    row.update({
        "id": str(segment["id"]),
        "doc_id": str(segment.get("doc_id", "")),
        "seg_id": str(segment.get("seg_id", "")),
        "source_lang": str(source_language.get("task_name", "en")),
        "target_lang": str(target["task_name"]),
        "target_lang_name": str(target["display_name"]),
        "model": adapter.model_name(),
        "model_key": model_key,
        "backend": adapter.backend_name(),
        "translation": translation,
        "timestamp": utc_timestamp(),
        "generation_params": {
            "batch_size": batch_size,
            "device_map": device_map,
            "do_sample": False,
            "max_input_length": max_input_length,
            "max_new_tokens": max_new_tokens,
        },
    })
    return row


def build_failure_row(
    segment: Dict[str, object],
    target: Dict[str, object],
    source_language: Dict[str, object],
    model_key: str,
) -> Dict[str, object]:
    row: Dict[str, object] = dict(segment)
    row.update({
        "id": str(segment["id"]),
        "doc_id": str(segment.get("doc_id", "")),
        "seg_id": str(segment.get("seg_id", "")),
        "source_lang": str(source_language.get("task_name", "en")),
        "target_lang": str(target["task_name"]),
        "target_lang_name": str(target["display_name"]),
        "model_key": model_key,
        "source_text": str(segment["seg"]),
        "timestamp": utc_timestamp(),
    })
    return row


def build_failure_record(
    segment: Dict[str, object],
    target: Dict[str, object],
    source_language: Dict[str, object],
    model_key: str,
    error: str,
) -> Dict[str, object]:
    return {
        "error": error,
        "row": build_failure_row(segment, target, source_language, model_key),
    }


def iter_failure_events(
    segments: Sequence[Dict[str, object]],
    target: Dict[str, object],
    source_language: Dict[str, object],
    model_key: str,
    error: str,
) -> Iterator[TranslationWriteEvent]:
    for segment in segments:
        yield TranslationWriteEvent(
            destination="failure",
            payload=build_failure_record(segment, target, source_language, model_key, error),
        )


def iter_translation_events(
    *,
    adapter: BaseAdapter,
    pending_segments: Sequence[Dict[str, object]],
    target: Dict[str, object],
    source_language: Dict[str, object],
    model_key: str,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    device_map: str,
) -> Iterator[TranslationWriteEvent]:
    for batch in iter_batches(pending_segments, batch_size):
        texts = [str(segment["seg"]) for segment in batch]
        try:
            translations = adapter.translate_batch(
                texts,
                target,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
            )
            if len(translations) != len(batch):
                raise RuntimeError(
                    f"expected {len(batch)} translations but received {len(translations)}"
                )
        except Exception as exc:
            error_text = f"batch translation failed: {exc}"
            yield from iter_failure_events(
                batch,
                target,
                source_language,
                model_key,
                error_text,
            )
            continue

        for segment, translation in zip(batch, translations):
            yield TranslationWriteEvent(
                destination="output",
                payload=build_output_row(
                    segment,
                    target,
                    source_language,
                    model_key,
                    adapter,
                    translation,
                    batch_size=batch_size,
                    max_input_length=max_input_length,
                    max_new_tokens=max_new_tokens,
                    device_map=device_map,
                ),
            )


def resolve_runtime_value(override: int | None, fallback: int) -> int:
    return override if override is not None else fallback


def command_validate(args: argparse.Namespace) -> int:
    context = collect_selection_context(args)
    print_selection_summary(context, args)

    if context["errors"]:
        print("Errors:")
        for error in context["errors"]:
            print(f"- {error}")
        return 2

    print("Validation passed.")
    return 0


def command_list_targets(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    registry_issues = validate_registry_schema(registry, args.registry)
    if registry_issues:
        for issue in registry_issues:
            print(issue)
        return 2

    languages = registry["languages"]
    for task_name, entry in languages.items():
        supported = ", ".join(entry["supported_by"]) if entry["supported_by"] else "none"
        print(f"{task_name}\t{entry['display_name']}\t{supported}")
    return 0


def command_execute(args: argparse.Namespace, *, mode_name: str) -> int:
    context = collect_selection_context(args)
    print_selection_summary(context, args)

    if context["errors"]:
        print("Errors:")
        for error in context["errors"]:
            print(f"- {error}")
        return 2

    runnable_targets = selected_plan_entries(context["execution_plan"], args.priority)
    if not runnable_targets:
        print("Errors:")
        print(f"- No runnable targets remain for priority selection '{args.priority}'.")
        return 2

    segments = load_source_segments(args.input, args.limit)
    if not segments:
        print("Errors:")
        print("- No non-empty input segments were selected for execution.")
        return 2

    print(f"Mode: {mode_name}")
    print(f"Priority selection: {args.priority}")
    print(f"Runnable target set: {format_plan_entries(runnable_targets)}")
    print(f"Selected segments: {len(segments)}")
    print(f"Resume enabled: {args.resume}")
    print(f"Plan only: {args.plan_only}")
    print(f"Output dir: {args.output_dir}")
    print(f"Log dir: {args.log_dir}")

    if args.plan_only:
        print("Planned jobs:")
        for target in runnable_targets:
            for model_key in target["eligible_models"]:
                output_path = output_path_for(args.output_dir, model_key, str(target["task_name"]))
                print(f"- {model_key} -> {target['task_name']} -> {output_path}")
        return 0

    registry = context["registry"]
    source_language = registry.get("source_language", {}) if isinstance(registry, dict) else {}
    adapter_cache: Dict[str, BaseAdapter] = {}
    adapter_load_errors: Dict[str, str] = {}
    had_execution_errors = False

    for target in runnable_targets:
        target_name = str(target["task_name"])
        for model_key in target["eligible_models"]:
            model_config = MODEL_CONFIGS[model_key]
            output_path = output_path_for(args.output_dir, model_key, target_name)
            failure_path = failure_path_for(args.log_dir, model_key, target_name)

            completed = load_completed_keys(output_path) if args.resume else set()
            pending_segments = [
                segment
                for segment in segments
                if (str(segment["id"]), target_name, model_config.hf_id) not in completed
            ]
            skipped_existing = len(segments) - len(pending_segments)
            batch_size = resolve_runtime_value(args.batch_size, model_config.default_batch_size)
            max_input_length = resolve_runtime_value(
                args.max_input_length,
                model_config.default_max_input_length,
            )
            max_new_tokens = resolve_runtime_value(
                args.max_new_tokens,
                model_config.default_max_new_tokens,
            )

            print(
                f"Job {model_key} -> {target_name}: total={len(segments)} pending={len(pending_segments)} resumed={skipped_existing}"
            )

            if not pending_segments:
                continue

            with TranslationRecordWriter(output_path, failure_path) as writer:
                if model_key in adapter_load_errors:
                    success_count, failure_count = writer.consume(
                        iter_failure_events(
                            pending_segments,
                            target,
                            source_language,
                            model_key,
                            adapter_load_errors[model_key],
                        )
                    )
                    had_execution_errors = had_execution_errors or failure_count > 0
                    print(
                        f"- completed {model_key} -> {target_name}: wrote={success_count} failures={failure_count}"
                    )
                    continue

                adapter = adapter_cache.get(model_key)
                if adapter is None:
                    try:
                        adapter = create_adapter(model_key, device_map=args.device_map)
                        adapter.load()
                        adapter_cache[model_key] = adapter
                    except Exception as exc:
                        error_text = f"adapter load failed: {exc}"
                        adapter_load_errors[model_key] = error_text
                        print(f"- {model_key} load failure: {exc}")
                        success_count, failure_count = writer.consume(
                            iter_failure_events(
                                pending_segments,
                                target,
                                source_language,
                                model_key,
                                error_text,
                            )
                        )
                        had_execution_errors = had_execution_errors or failure_count > 0
                        print(
                            f"- completed {model_key} -> {target_name}: wrote={success_count} failures={failure_count}"
                        )
                        continue

                success_count, failure_count = writer.consume(
                    iter_translation_events(
                        adapter=adapter,
                        pending_segments=pending_segments,
                        target=target,
                        source_language=source_language,
                        model_key=model_key,
                        batch_size=batch_size,
                        max_input_length=max_input_length,
                        max_new_tokens=max_new_tokens,
                        device_map=str(args.device_map),
                    )
                )
                had_execution_errors = had_execution_errors or failure_count > 0
                print(
                    f"- completed {model_key} -> {target_name}: wrote={success_count} failures={failure_count}"
                )

    if had_execution_errors:
        print("Execution completed with failures logged to the log directory.")
        return 2

    print("Execution completed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return command_validate(args)
    if args.command == "list-targets":
        return command_list_targets(args)
    if args.command == "smoke-test":
        return command_execute(args, mode_name="smoke-test")
    if args.command == "run":
        return command_execute(args, mode_name="run")

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())