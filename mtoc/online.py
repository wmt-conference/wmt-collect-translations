from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

from interfaces import BaseBackend, RuntimeModelConfig, TranslationRequest, TranslationResult


ProviderCallable = Callable[[Dict[str, object]], object]


class OnlineProviderBackend(BaseBackend):
    """Adapter boundary for API/online providers.

    This class intentionally does not import any provider implementation. It can
    wrap a colleague's existing callable later as long as that callable accepts a
    request dictionary and returns either a string or ``(text, metadata)``.
    """

    def __init__(self, model_config: RuntimeModelConfig, provider: ProviderCallable) -> None:
        super().__init__(model_config)
        self.provider = provider

    def backend_name(self) -> str:
        return "online"

    def model_name(self) -> str:
        return self.model_config.model_id

    def load(self) -> None:
        return None

    def translate_batch(
        self,
        requests: Sequence[TranslationRequest],
        *,
        max_input_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[TranslationResult]:
        results: List[TranslationResult] = []
        for request in requests:
            provider_request = self._build_provider_request(
                request,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            raw_result = self.provider(provider_request)
            translation, metadata = self._normalize_provider_result(raw_result)
            results.append(TranslationResult(
                request=request,
                translation=translation,
                model_key=self.model_config.key,
                model_id=self.model_config.model_id,
                backend="online",
                metadata=metadata,
            ))
        return results

    def _build_provider_request(
        self,
        request: TranslationRequest,
        *,
        max_input_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Dict[str, object]:
        provider_request = dict(request.raw)
        provider_request.update({
            "doc_id": request.doc_id,
            "source_doc": request.source_doc,
            "tgt_lang": request.tgt_lang,
            "instruction": request.instruction,
            "prompt": request.prompt(),
            "max_input_length": max_input_length,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        })
        return provider_request

    def _normalize_provider_result(self, raw_result: object) -> tuple[str, Dict[str, object]]:
        if raw_result is None:
            return "", {"provider_result": None}
        if isinstance(raw_result, tuple) and raw_result:
            text = str(raw_result[0]) if raw_result[0] is not None else ""
            metadata = raw_result[1] if len(raw_result) > 1 and isinstance(raw_result[1], dict) else {}
            return text, metadata
        return str(raw_result), {}
