from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class RuntimeModelConfig:
    key: str
    model_id: str
    backend: str
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    default_batch_size: int = 1
    default_max_input_length: int = 4096
    default_max_new_tokens: int = 1024
    trust_remote_code: bool = False
    gated: bool = False
    track: str = ""
    run_scope: str = ""
    notes: str = ""


@dataclass(frozen=True)
class TranslationRequest:
    doc_id: str
    source_doc: str
    tgt_lang: str
    instruction: str
    raw: Dict[str, object] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str]:
        return self.doc_id, self.tgt_lang

    def prompt(self) -> str:
        instruction = self.instruction.strip()
        source_doc = self.source_doc.strip()
        if not instruction:
            return source_doc
        if not source_doc:
            return instruction
        return f"{instruction}\n\n{source_doc}"


@dataclass(frozen=True)
class TranslationResult:
    request: TranslationRequest
    translation: str
    model_key: str
    model_id: str
    backend: str
    metadata: Dict[str, object] = field(default_factory=dict)


class BaseBackend:
    def __init__(self, model_config: RuntimeModelConfig) -> None:
        self.model_config = model_config

    def backend_name(self) -> str:
        return self.model_config.backend

    def model_name(self) -> str:
        return self.model_config.model_id

    def load(self) -> None:
        raise NotImplementedError()

    def translate_batch(
        self,
        requests: Sequence[TranslationRequest],
        *,
        max_input_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[TranslationResult]:
        raise NotImplementedError()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
