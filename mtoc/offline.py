from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from interfaces import BaseBackend, RuntimeModelConfig, TranslationRequest, TranslationResult


def _is_gpt_oss(model_config: RuntimeModelConfig) -> bool:
    return "gpt-oss" in model_config.model_id.lower() or "gpt_oss" in model_config.key.lower()


def _apply_chat_template(tokenizer: Any, messages: List[Dict[str, str]], *, tokenize: bool, return_dict: bool, gpt_oss: bool) -> Any:
    """Apply a chat template, suppressing reasoning where the model supports it.

    gpt-oss honours ``reasoning_effort`` (Harmony format); Qwen-style models honour
    ``enable_thinking=False``. Both are passed opportunistically and we degrade to a
    plain template when the tokenizer rejects an unknown kwarg.
    """
    base: Dict[str, Any] = {"add_generation_prompt": True}
    if return_dict:
        base["return_dict"] = True
    attempts: List[Dict[str, Any]] = []
    if gpt_oss:
        attempts.append({**base, "reasoning_effort": "low"})
    attempts.append({**base, "enable_thinking": False})
    attempts.append(base)
    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=tokenize, **kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return tokenizer.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=True)


def _extract_gpt_oss_final(text: str) -> str:
    """Return only the Harmony ``final`` channel from a gpt-oss completion.

    gpt-oss emits ``analysis<reasoning>...assistantfinal<answer>``. We keep just the
    answer. If the model never reached the final channel (e.g. it exhausted the token
    budget mid-analysis), return an empty string so the row is flagged as a failure
    rather than polluted with raw reasoning text.
    """
    marker = "assistantfinal"
    index = text.rfind(marker)
    if index != -1:
        return text[index + len(marker):].strip()
    # Fallback: some renderings expose the channel as a bare ``final`` tag.
    final_index = text.rfind("final")
    if final_index != -1 and "analysis" in text[:final_index]:
        return text[final_index + len("final"):].strip()
    return ""



def create_offline_backend(
    model_config: RuntimeModelConfig,
    *,
    backend_override: str | None = None,
    device_map: str = "auto",
    gpu_memory_utilization: float = 0.9,
    max_memory_per_gpu: str | None = None,
    allow_cpu_offload: bool = False,
) -> BaseBackend:
    backend = backend_override if backend_override and backend_override != "registry" else model_config.backend
    if backend == "hf":
        return HfCausalLmAdapter(
            model_config,
            device_map=device_map,
            gpu_memory_utilization=gpu_memory_utilization,
            max_memory_per_gpu=max_memory_per_gpu,
            allow_cpu_offload=allow_cpu_offload,
        )
    if backend == "vllm":
        return VllmAdapter(model_config, gpu_memory_utilization=gpu_memory_utilization)
    if backend == "auto":
        return AutoBackend(
            model_config,
            device_map=device_map,
            gpu_memory_utilization=gpu_memory_utilization,
            max_memory_per_gpu=max_memory_per_gpu,
            allow_cpu_offload=allow_cpu_offload,
        )
    raise ValueError(f"Unsupported backend for {model_config.key}: {backend}")


class HfCausalLmAdapter(BaseBackend):
    def __init__(
        self,
        model_config: RuntimeModelConfig,
        *,
        device_map: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_memory_per_gpu: str | None = None,
        allow_cpu_offload: bool = False,
    ) -> None:
        super().__init__(model_config)
        self.device_map = device_map
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_memory_per_gpu = max_memory_per_gpu
        self.allow_cpu_offload = allow_cpu_offload
        self.model: Any = None
        self.tokenizer: Any = None
        self.torch: Any = None
        self._use_cache: bool = True
        self._uses_chat_template: bool = False
        self._is_gpt_oss: bool = _is_gpt_oss(model_config)

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for --backend hf") from exc

        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for --backend hf") from exc

        self.torch = torch
        self._ensure_remote_code_compat()
        dtype = self._resolve_torch_dtype()
        model_path = self._resolve_model_path()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Mistral's MistralCommonBackend exposes apply_chat_template but no
        # chat_template attribute, so detect it explicitly.
        self._uses_chat_template = (
            getattr(self.tokenizer, "chat_template", None) is not None
            or type(self.tokenizer).__name__ == "MistralCommonBackend"
        )

        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        # Some remote-code configs (e.g. ChatGLM) require ``max_length`` during
        # model init but it must not leak into generation as a forced length.
        injected_max_length = False
        if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
            config.max_length = config.seq_length
            injected_max_length = True

        # Upcast FP8 checkpoints to the compute dtype (bf16) so they run without
        # the optional fine-grained FP8 kernels.
        self._enable_fp8_dequantize(config)

        model_class = AutoModelForImageTextToText if config.model_type == "mistral3" else AutoModelForCausalLM
        device_map = None if config.model_type == "chatglm" else self.device_map
        # Legacy ChatGLM remote code predates the Transformers cache API.
        self._use_cache = config.model_type != "chatglm"

        from_pretrained_kwargs: Dict[str, Any] = {
            "config": config,
            "device_map": device_map,
            "torch_dtype": dtype,
            "trust_remote_code": self.model_config.trust_remote_code,
        }
        # Cap per-GPU memory so Accelerate shards across the visible GPUs and
        # never silently offloads layers to CPU/disk (which destroys throughput).
        max_memory = self._build_max_memory()
        if device_map is not None and max_memory is not None:
            from_pretrained_kwargs["max_memory"] = max_memory

        self.model = model_class.from_pretrained(model_path, **from_pretrained_kwargs)
        if device_map is None and self.torch.cuda.is_available():
            self.model.to("cuda")
        self._verify_device_placement()
        if injected_max_length:
            # Remove the init-only attribute so Transformers does not treat it as
            # a forced generation length; only ``max_new_tokens`` controls decoding.
            try:
                delattr(self.model.config, "max_length")
            except AttributeError:
                pass
            generation_config = getattr(self.model, "generation_config", None)
            if generation_config is not None:
                generation_config.max_length = None
        self.model.eval()

    def translate_batch(
        self,
        requests: Sequence[TranslationRequest],
        *,
        max_input_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[TranslationResult]:
        self.load()
        encoded = self._encode_requests(requests, max_input_length)
        encoded = self._move_inputs(encoded)
        prompt_width = encoded["input_ids"].shape[1]

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": self._use_cache,
            # Curb runaway repetition/paraphrase loops on greedy decoding.
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 4,
        }
        eos_token_id = self._resolve_eos_token_id()
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            generation_kwargs.update({"do_sample": False})

        with self.torch.inference_mode():
            generated = self.model.generate(**encoded, **generation_kwargs)

        results: List[TranslationResult] = []
        for request, row in zip(requests, generated):
            new_tokens = row[prompt_width:]
            translation = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if self._is_gpt_oss:
                translation = _extract_gpt_oss_final(translation)
            results.append(TranslationResult(
                request=request,
                translation=translation,
                model_key=self.model_config.key,
                model_id=self.model_config.model_id,
                backend="hf",
            ))
        return results

    def _encode_requests(self, requests: Sequence[TranslationRequest], max_input_length: int) -> Dict[str, Any]:
        # Prefer direct chat-template tokenization (token ids) over rendering to a
        # string and re-encoding. The string round-trip mangles special tokens for
        # backends like MistralCommon, producing literal <s>/[INST] artifacts and
        # occasional empty generations.
        if self._uses_chat_template:
            token_id_lists = [
                self._chat_template_ids(request.prompt(), max_input_length) for request in requests
            ]
            return self._left_pad(token_id_lists)
        prompts = [request.prompt() for request in requests]
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )

    def _chat_template_ids(self, prompt: str, max_input_length: int) -> List[int]:
        messages = [{"role": "user", "content": prompt}]
        encoded = _apply_chat_template(
            self.tokenizer, messages, tokenize=True, return_dict=True, gpt_oss=self._is_gpt_oss
        )
        ids = list(encoded["input_ids"])
        if max_input_length and len(ids) > max_input_length:
            # Keep the tail so the generation prompt (e.g. [/INST]) is preserved.
            ids = ids[-max_input_length:]
        return ids

    def _left_pad(self, token_id_lists: Sequence[Sequence[int]]) -> Dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        max_len = max((len(ids) for ids in token_id_lists), default=1)
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for ids in token_id_lists:
            pad_len = max_len - len(ids)
            input_ids.append([pad_id] * pad_len + list(ids))
            attention_mask.append([0] * pad_len + [1] * len(ids))
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(attention_mask, dtype=self.torch.long),
        }

    def _resolve_eos_token_id(self) -> Any:
        # Prefer the model's generation config, which carries the full stop-token
        # set (e.g. ChatGLM uses three EOS ids); fall back to the tokenizer.
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            eos = getattr(generation_config, "eos_token_id", None)
            if eos is not None:
                return eos
        return getattr(self.tokenizer, "eos_token_id", None)

    @staticmethod
    def _ensure_remote_code_compat() -> None:
        # Older remote-code model classes (e.g. ChatGLM) predate
        # ``all_tied_weights_keys`` that recent Transformers expects during
        # loading. A class-level empty default is shadowed by the per-instance
        # attribute that natively supported models set, so this is harmless.
        try:
            from transformers.modeling_utils import PreTrainedModel
        except ImportError:
            return
        import inspect

        if inspect.getattr_static(PreTrainedModel, "all_tied_weights_keys", None) is None:
            PreTrainedModel.all_tied_weights_keys = {}

    @staticmethod
    def _enable_fp8_dequantize(config: Any) -> None:
        quant_config = getattr(config, "quantization_config", None)
        if quant_config is None:
            return

        def is_fp8(method: Any) -> bool:
            return "fp8" in str(method).lower()

        if isinstance(quant_config, dict):
            if is_fp8(quant_config.get("quant_method")):
                quant_config["dequantize"] = True
        elif is_fp8(getattr(quant_config, "quant_method", None)):
            setattr(quant_config, "dequantize", True)

    def _resolve_torch_dtype(self) -> Any:
        dtype_name = self.model_config.dtype.lower()
        if dtype_name == "bfloat16":
            return self.torch.bfloat16
        if dtype_name == "float16":
            return self.torch.float16
        if dtype_name == "float32":
            return self.torch.float32
        if dtype_name == "auto":
            return "auto"
        raise ValueError(f"Unsupported torch dtype: {self.model_config.dtype}")

    def _build_max_memory(self) -> Dict[Any, Any] | None:
        if not self.torch.cuda.is_available():
            return None
        device_count = self.torch.cuda.device_count()
        if device_count == 0:
            return None

        max_memory: Dict[Any, Any] = {}
        for index in range(device_count):
            if self.max_memory_per_gpu:
                max_memory[index] = self.max_memory_per_gpu
            else:
                total = self.torch.cuda.get_device_properties(index).total_memory
                max_memory[index] = int(total * self.gpu_memory_utilization)
        if not self.allow_cpu_offload:
            # Forbid CPU/disk offload so an oversized model fails loudly instead
            # of running at a crawl.
            max_memory["cpu"] = 0
        return max_memory

    def _verify_device_placement(self) -> None:
        device_map = getattr(self.model, "hf_device_map", None)
        if not device_map:
            return
        placements = set(str(device) for device in device_map.values())
        offloaded = {device for device in placements if device in {"cpu", "disk"}}
        gpu_devices = sorted(device for device in placements if device not in {"cpu", "disk"})
        if offloaded and not self.allow_cpu_offload:
            raise RuntimeError(
                f"{self.model_config.key}: model does not fit on the visible GPUs and "
                f"layers were offloaded to {sorted(offloaded)}. Provide more GPUs "
                f"(increase tensor_parallel_size / CUDA_VISIBLE_DEVICES) or pass "
                f"--allow-cpu-offload to permit slow offloaded inference."
            )
        print(
            f"{self.model_config.key}: HF device placement across {len(gpu_devices)} GPU(s): "
            f"{', '.join(gpu_devices) if gpu_devices else 'single device'}",
            flush=True,
        )

    def _resolve_model_path(self) -> str:
        if self.model_config.trust_remote_code:
            return self.model_config.model_id
        cache_root = os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE")
        if not cache_root:
            return self.model_config.model_id
        cache_path = Path(cache_root) / ("models--" + self.model_config.model_id.replace("/", "--")) / "snapshots"
        if not cache_path.is_dir():
            return self.model_config.model_id
        snapshots = sorted(path for path in cache_path.iterdir() if path.is_dir())
        if not snapshots:
            return self.model_config.model_id
        return str(snapshots[-1])

    def _move_inputs(self, encoded: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            return encoded
        try:
            first_parameter = next(self.model.parameters())
        except StopIteration:
            return encoded
        device = first_parameter.device
        if str(device) == "cpu":
            return encoded
        return {key: value.to(device) for key, value in encoded.items()}


class VllmAdapter(BaseBackend):
    def __init__(self, model_config: RuntimeModelConfig, *, gpu_memory_utilization: float) -> None:
        super().__init__(model_config)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm: Any = None
        self.tokenizer: Any = None
        self._is_gpt_oss: bool = _is_gpt_oss(model_config)
        self._is_mistral: bool = "mistral" in model_config.model_id.lower()

    def load(self) -> None:
        if self.llm is not None:
            return

        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("vllm is required for --backend vllm") from exc

        llm_kwargs: Dict[str, Any] = dict(
            model=self.model_config.model_id,
            tensor_parallel_size=self.model_config.tensor_parallel_size,
            dtype=self.model_config.dtype,
            max_model_len=self.model_config.default_max_input_length,
            trust_remote_code=self.model_config.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        if self._is_mistral:
            # Mistral vision-language checkpoints (e.g. Mistral3ForConditionalGeneration,
            # Ministral) fail vLLM's multimodal profiling because the MistralCommon
            # tokenizer injects a dummy image. This is a text translation task, so use
            # the native Mistral tokenizer and forbid image inputs, which lets vLLM
            # serve the (FP8) text model instead of falling back to slow HF.
            llm_kwargs["tokenizer_mode"] = "mistral"
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0}

        try:
            self.llm = LLM(**llm_kwargs)
        except TypeError:
            # Older vLLM builds may not accept limit_mm_per_prompt as a dict here.
            llm_kwargs.pop("limit_mm_per_prompt", None)
            self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def translate_batch(
        self,
        requests: Sequence[TranslationRequest],
        *,
        max_input_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[TranslationResult]:
        self.load()
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError("vllm is required for --backend vllm") from exc

        prompts = [request.prompt() for request in requests]
        rendered_prompts = [self._render_prompt(prompt) for prompt in prompts]
        # Anti-repetition defaults: break the degenerate reasoning loops gpt-oss
        # falls into on low-resource target languages (which otherwise exhaust the
        # token budget before emitting a final answer). repetition_penalty curbs
        # token reuse and frequency_penalty ramps up as a token repeats, so severe
        # loops are suppressed while normal translation is barely affected. vLLM
        # has no no_repeat_ngram_size, so these penalties stand in for the HF path.
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            frequency_penalty=0.3,
        )
        outputs = self.llm.generate(rendered_prompts, sampling_params)
        results: List[TranslationResult] = []
        for request, output in zip(requests, outputs):
            translation = output.outputs[0].text.strip() if output.outputs else ""
            if self._is_gpt_oss:
                translation = _extract_gpt_oss_final(translation)
            results.append(TranslationResult(
                request=request,
                translation=translation,
                model_key=self.model_config.key,
                model_id=self.model_config.model_id,
                backend="vllm",
            ))
        return results

    def _render_prompt(self, prompt: str) -> str:
        # Mistral tokenizers (tokenizer_mode="mistral") expose apply_chat_template
        # but may not surface a ``chat_template`` attribute, so detect them by name
        # as well and attempt templating regardless.
        has_template = getattr(self.tokenizer, "chat_template", None) is not None
        is_mistral_tok = "mistral" in type(self.tokenizer).__name__.lower() or self._is_mistral
        if has_template or is_mistral_tok:
            messages = [{"role": "user", "content": prompt}]
            try:
                return _apply_chat_template(
                    self.tokenizer, messages, tokenize=False, return_dict=False, gpt_oss=self._is_gpt_oss
                )
            except Exception:
                return prompt
        return prompt


class AutoBackend(BaseBackend):
    """Prefer vLLM, fall back to the HF adapter if vLLM cannot load the model.

    vLLM is the faster runtime and handles quantized checkpoints (e.g. gpt-oss
    MXFP4) that the HF loader chokes on, but it does not support every
    architecture (legacy remote-code models, some vision-language classes). This
    wrapper tries vLLM first and transparently falls back to Transformers/HF so a
    single registry default works across the whole model set.
    """

    def __init__(
        self,
        model_config: RuntimeModelConfig,
        *,
        device_map: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_memory_per_gpu: str | None = None,
        allow_cpu_offload: bool = False,
    ) -> None:
        super().__init__(model_config)
        self._gpu_memory_utilization = gpu_memory_utilization
        self._hf_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_memory_per_gpu": max_memory_per_gpu,
            "allow_cpu_offload": allow_cpu_offload,
        }
        self.delegate: BaseBackend | None = None

    def load(self) -> None:
        if self.delegate is not None:
            return
        try:
            adapter: BaseBackend = VllmAdapter(
                self.model_config, gpu_memory_utilization=self._gpu_memory_utilization
            )
            adapter.load()
        except Exception as exc:  # noqa: BLE001 - any vLLM failure should fall back
            print(
                f"{self.model_config.key}: vLLM backend unavailable ({type(exc).__name__}: {exc}); "
                f"falling back to HF backend",
                flush=True,
            )
            self._free_cuda()
            adapter = HfCausalLmAdapter(self.model_config, **self._hf_kwargs)
            adapter.load()
            print(f"{self.model_config.key}: using HF backend", flush=True)
        else:
            print(f"{self.model_config.key}: using vLLM backend", flush=True)
        self.delegate = adapter

    def translate_batch(
        self,
        requests: Sequence[TranslationRequest],
        *,
        max_input_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[TranslationResult]:
        self.load()
        assert self.delegate is not None
        return self.delegate.translate_batch(
            requests,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    @staticmethod
    def _free_cuda() -> None:
        # Best-effort release of any memory a failed vLLM init left behind before
        # the HF loader allocates the model.
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must never mask the real error
            pass
