from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DecodingConfig:
    """A named decoding/generation variant for an experiment.

    ``method`` selects the search strategy: ``greedy`` (argmax), ``sample``
    (temperature/top-p nucleus sampling) or ``beam`` (beam search). ``thinking``
    toggles a model's reasoning channel where supported (e.g. Qwen
    ``enable_thinking``, gpt-oss ``reasoning_effort``); ``None`` keeps the safe
    default (reasoning suppressed). ``max_new_tokens`` optionally overrides the
    per-model generation length for this variant. ``system_prompt`` optionally
    prepends a ``system``-role message (native on models like Gemma 4) to steer
    the model, e.g. to constrain a talky instruct model to output only the
    translation. It falls back to a folded-in user prefix on models whose chat
    template rejects a system role.
    """

    name: str = "default"
    method: str = "greedy"
    temperature: float = 0.0
    top_p: float = 1.0
    num_beams: int = 1
    length_penalty: float = 1.0
    thinking: Optional[bool] = None
    seed: Optional[int] = None
    max_new_tokens: Optional[int] = None
    system_prompt: Optional[str] = None


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
    variants: Tuple[DecodingConfig, ...] = field(default_factory=tuple)



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
    # Whether the backend schedules its own continuous batching (vLLM). When True,
    # the runner hands it large prompt chunks at once instead of tiny synchronous
    # micro-batches, so the engine can keep the GPU saturated.
    continuous_batching: bool = False

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
        decoding: "DecodingConfig",
    ) -> List[TranslationResult]:
        raise NotImplementedError()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
