import os
import logging
from tools.cache import get_cache, cache_key
from tools.errors import FINISH_LENGTH, FINISH_STOP

MODELS = {
    "DeepSeek-V4-Pro": {"api_model": "deepseek-ai/DeepSeek-V4-Pro", "extra": {"max_tokens": 32768, "temperature": 1.0, "top_p": 1.0, "reasoning_effort": "high", "reasoning": {"enabled": True}}},
}

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from together import Together
        assert "TOGETHER_API_KEY" in os.environ, "Please set the TOGETHER_API_KEY environment variable"
        CLIENT = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    return CLIENT


def process(request, model, extra=None, api_model=None):
    extra = extra or {}
    api_model = api_model or model
    cache = get_cache("together_ai")
    key = cache_key(model, request)

    if key in cache:
        raw, extra = cache[key]["raw"], cache[key]["extra"]
    else:
        raw = _call(request, api_model, extra)
        if raw is None:
            return None
        cache[key] = {"raw": raw, "extra": extra}

    return _extract(raw, extra)


def _call(request, model, extra):
    import together

    client = lazy_get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request['prompt']}],
            **extra,
        )
    except together.error.APIError as e:
        print(f"APIError: {e}")
        return None

    return response.model_dump(mode="json")


def _extract(raw, extra):
    if raw['choices'][0]['finish_reason'] == "length":
        finish_reason = FINISH_LENGTH
    elif raw['choices'][0]['finish_reason'] == "stop":
        finish_reason = FINISH_STOP
    else:
        logging.warning(f"Finish reason: {raw['choices'][0]['finish_reason']}; {raw['choices'][0]['message']['content']}")
        return None

    message = raw['choices'][0]['message']
    completion_details = raw['usage'].get('completion_tokens_details') or {}

    return message['content'], {
        "raw_response": raw,
        "model": raw['model'],
        "extra": extra,
        "reasoning_trace": message.get('reasoning') or message.get('reasoning_content') or None,
        "input_tokens": raw['usage']['prompt_tokens'],
        "output_tokens": raw['usage']['completion_tokens'],
        "thinking_tokens": completion_details.get('reasoning_tokens', 0),
        "finish_reason": finish_reason
    }
