from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from interfaces import BaseBackend, RuntimeModelConfig, TranslationRequest, TranslationResult


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
        try:
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        ids = list(ids)
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

    def load(self) -> None:
        if self.llm is not None:
            return

        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("vllm is required for --backend vllm") from exc

        self.llm = LLM(
            model=self.model_config.model_id,
            tensor_parallel_size=self.model_config.tensor_parallel_size,
            dtype=self.model_config.dtype,
            max_model_len=self.model_config.default_max_input_length,
            trust_remote_code=self.model_config.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
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
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        outputs = self.llm.generate(rendered_prompts, sampling_params)
        results: List[TranslationResult] = []
        for request, output in zip(requests, outputs):
            translation = output.outputs[0].text.strip() if output.outputs else ""
            results.append(TranslationResult(
                request=request,
                translation=translation,
                model_key=self.model_config.key,
                model_id=self.model_config.model_id,
                backend="vllm",
            ))
        return results

    def _render_prompt(self, prompt: str) -> str:
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return prompt
