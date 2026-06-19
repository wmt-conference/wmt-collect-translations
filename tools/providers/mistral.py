import os
import logging
from tools.cache import get_cache, cache_key
from tools.errors import FINISH_LENGTH, FINISH_STOP

MODELS = {
    "mistral-medium-3.5": {"max_tokens": 8192, "temperature": 0.0},
}

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from mistralai import Mistral

        assert "MISTRAL_API_KEY" in os.environ, "Please set the MISTRAL_API_KEY environment variable"

        CLIENT = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    return CLIENT


def process(request, model, max_tokens, temperature):
    cache = get_cache("mistral")
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
    client = lazy_get_client()

    messages = [{"role": "user", "content": request['prompt']}]

    try:
        response = client.chat.complete(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

    return response.model_dump(mode="json")


def _extract(raw, temperature):
    if raw['choices'][0]['finish_reason'] == "stop":
        finish_reason = FINISH_STOP
    elif raw['choices'][0]['finish_reason'] == "length":
        finish_reason = FINISH_LENGTH
    else:
        return None

    return raw['choices'][0]['message']['content'], {
        "raw_response": raw,
        "model": raw['model'],
        "temperature": temperature,
        "reasoning_trace": None,
        "input_tokens": raw['usage']['prompt_tokens'],
        "output_tokens": raw['usage']['completion_tokens'],
        "thinking_tokens": 0,
        "finish_reason": finish_reason
    }
