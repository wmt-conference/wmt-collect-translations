from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple


ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\u200b\u200c\u200d\ufeff\u2060")


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSONL row: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield line_number, payload


def iter_segments(path: Path) -> Iterator[Dict[str, object]]:
    for _, payload in iter_jsonl(path):
        yield payload


def normalize_source_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.translate(ZERO_WIDTH_TRANSLATION).strip()


def validate_input_schema(path: Path, required_fields: Sequence[str]) -> Dict[str, object]:
    row_count = 0
    sample_keys: Tuple[str, ...] = ()
    issues: List[str] = []

    for line_number, payload in iter_jsonl(path):
        row_count += 1
        if not sample_keys:
            sample_keys = tuple(payload.keys())

        missing = [field for field in required_fields if field not in payload]
        if missing:
            payload_id = payload.get("id", f"line {line_number}")
            issues.append(
                f"row {line_number} ({payload_id}) is missing required fields: {', '.join(missing)}"
            )

    if row_count == 0:
        issues.append("input contains no JSONL records")

    return {
        "row_count": row_count,
        "sample_keys": sample_keys,
        "issues": issues,
    }


def load_completed_keys(output_path: Path) -> Set[Tuple[str, str, str]]:
    if not output_path.exists():
        return set()

    completed: Set[Tuple[str, str, str]] = set()
    for _, payload in iter_jsonl(output_path):
        record_id = str(payload.get("id", ""))
        target_lang = str(payload.get("target_lang", ""))
        model_name = str(payload.get("model", ""))
        if record_id and target_lang and model_name:
            completed.add((record_id, target_lang, model_name))
    return completed


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def write(self, row: Dict[str, object]) -> None:
        handle = self._ensure_open()
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def write_all(self, rows: Iterable[Dict[str, object]]) -> None:
        for row in rows:
            self.write(row)

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None

    def _ensure_open(self):
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        return self._handle