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
) -> BaseBackend:
    backend = backend_override if backend_override and backend_override != "registry" else model_config.backend
    if backend == "hf":
        return HfCausalLmAdapter(model_config, device_map=device_map)
    if backend == "vllm":
        return VllmAdapter(model_config, gpu_memory_utilization=gpu_memory_utilization)
    raise ValueError(f"Unsupported backend for {model_config.key}: {backend}")


class HfCausalLmAdapter(BaseBackend):
    def __init__(self, model_config: RuntimeModelConfig, *, device_map: str = "auto") -> None:
        super().__init__(model_config)
        self.device_map = device_map
        self.model: Any = None
        self.tokenizer: Any = None
        self.torch: Any = None

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
        dtype = self._resolve_torch_dtype()
        model_path = self._resolve_model_path()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
            config.max_length = config.seq_length

        model_class = AutoModelForImageTextToText if config.model_type == "mistral3" else AutoModelForCausalLM
        device_map = None if config.model_type == "chatglm" else self.device_map
        self.model = model_class.from_pretrained(
            model_path,
            config=config,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if device_map is None and self.torch.cuda.is_available():
            self.model.to("cuda")
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
        prompts = [request.prompt() for request in requests]
        rendered_prompts = [self._render_prompt(prompt) for prompt in prompts]
        encoded = self.tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        encoded = self._move_inputs(encoded)
        prompt_width = encoded["input_ids"].shape[1]

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
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
