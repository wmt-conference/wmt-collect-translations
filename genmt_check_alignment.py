import argparse
import json
import re
from pathlib import Path
import ipdb

LOG_FILE = Path("alignment.log")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR / "wmt26_genmt_blindset.jsonl"
DEFAULT_TRANSLATION = SCRIPT_DIR / "translations.jsonl"


def load_jsonl(path: Path) -> dict[str, dict]:
    entries = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries[entry["doc_id"]] = entry
    return entries


def parse_json(text: str):
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", text))


def aligned_json(src_text: str, tgt_text: str):
    src_data = parse_json(src_text)
    src_count = len(src_data) if isinstance(src_data, (dict, list)) else 1
    try:
        tgt_data = parse_json(tgt_text)
        tgt_count = len(tgt_data) if isinstance(tgt_data, (dict, list)) else 1
    except json.JSONDecodeError:
        tgt_count = "invalid_json"
    return src_count, tgt_count


def aligned_html(src_text: str, tgt_text: str):
    src_count = src_text.count("<p>")
    tgt_count = tgt_text.count("<p>")
    return src_count, tgt_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path", type=Path, default=DEFAULT_SOURCE,
        help=f"Path to the source .jsonl file (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--translation_path", type=Path, default=DEFAULT_TRANSLATION,
        help=f"Path to the translation .jsonl file (default: {DEFAULT_TRANSLATION})",
    )
    args = parser.parse_args()

    source_path, translation_path = args.source_path, args.translation_path

    if not source_path.is_file() or not translation_path.is_file():
        parser.error(f"SOURCE_JSONL or TRANSLATION_JSONL does not exist:\n  {source_path}\n  {translation_path}")

    source_docs = load_jsonl(source_path)
    target_docs = load_jsonl(translation_path)

    extra_in_translation = sorted(set(target_docs) - set(source_docs))
    if extra_in_translation:
        raise ValueError(
            f"{len(extra_in_translation)} doc_id(s) in {translation_path} are missing from {source_path}: "
            f"{extra_in_translation}"
        )

    common = sorted(set(source_docs) & set(target_docs))
    missing_in_translation = sorted(set(source_docs) - set(target_docs))

    log_lines = []
    checked = aligned = misaligned = skipped = 0

    for doc_id in common:
        src_text = source_docs[doc_id]["source_doc"].strip()
        tgt_text = target_docs[doc_id]["hypothesis"].strip()

        if src_text.startswith("```json"):
            doc_type = "json"
            src_count, tgt_count = aligned_json(src_text, tgt_text)
        elif src_text.startswith("<p>"):
            doc_type = "html"
            src_count, tgt_count = aligned_html(src_text, tgt_text)
        else:
            skipped += 1
            continue

        checked += 1
        if src_count == tgt_count:
            aligned += 1
            continue

        misaligned += 1
        line = f"MISALIGNED {doc_id}: {doc_type} source={src_count} target={tgt_count}"
        print(line)
        log_lines.append(line)

    summary = (
        f"{checked} checked, {aligned} aligned, {misaligned} misaligned, "
        f"{skipped} skipped ({len(missing_in_translation)} missing from translation file)"
    )
    print(summary)
    log_lines.append(summary)

    LOG_FILE.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"\nWrote log to {LOG_FILE}")


if __name__ == "__main__":
    main()
