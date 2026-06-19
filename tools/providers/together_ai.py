import os
import logging
from tools.cache import get_cache, cache_key
from tools.errors import FINISH_LENGTH, FINISH_STOP

MODELS = {
    "deepseek-ai/DeepSeek-V3": {"max_tokens": 8192, "temperature": None},
    "Qwen/Qwen3-235B-A22B-fp8-tput": {"max_tokens": 8192, "temperature": None},
    "Qwen/Qwen2.5-7B-Instruct-Turbo": {"max_tokens": 8192, "temperature": None},
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {"max_tokens": 8192, "temperature": None},
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {"max_tokens": 8192, "temperature": None},
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {"max_tokens": 8192, "temperature": None},
    "mistralai/Mistral-7B-Instruct-v0.3": {"max_tokens": 8192, "temperature": None},
}

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from together import Together
        assert "TOGETHER_API_KEY" in os.environ, "Please set the TOGETHER_API_KEY environment variable"
        CLIENT = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    return CLIENT


def process(request, model, max_tokens, temperature):
    cache = get_cache("together_ai")
    key = cache_key(model, request)

    if key in cache:
        raw = cache[key]
    else:
        raw = _call(request, model, max_tokens, temperature)
        if raw is None:
            return None
        cache[key] = raw

    return _extract(raw, temperature)


def _call(request, model, max_tokens, temperature):
    import together

    client = lazy_get_client()
    extra = {"temperature": temperature} if temperature is not None else {}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request['prompt']}],
            max_tokens=max_tokens,
            chat_template_kwargs={
                "enable_thinking": False, # turns off QWEN thinking
                "max_tokens": max_tokens,
                **extra,
            },
            **extra,
        )
    except together.error.APIError as e:
        print(f"APIError: {e}")
        return None

    return response.model_dump(mode="json")


def _extract(raw, temperature):
    if raw['choices'][0]['finish_reason'] == "length":
        finish_reason = FINISH_LENGTH
    elif raw['choices'][0]['finish_reason'] == "stop":
        finish_reason = FINISH_STOP
    else:
        logging.warning(f"Finish reason: {raw['choices'][0]['finish_reason']}; {raw['choices'][0]['message']['content']}")
        return None

    return raw['choices'][0]['message']['content'], {
        "raw_response": raw,
        "model": raw['model'],
        "temperature": temperature,
        "reasoning_trace": None,
        "input_tokens": raw['usage']['prompt_tokens'],
        "output_tokens": raw['usage']['completion_tokens'],
        "thinking_tokens": 0,  # Together AI does not provide thinking tokens
        "finish_reason": finish_reason
    }
