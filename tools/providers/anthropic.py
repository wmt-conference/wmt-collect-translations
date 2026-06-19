import os
import logging
from tools.cache import get_cache, cache_key
from tools.errors import FINISH_STOP, FINISH_LENGTH

MODELS = {
    "claude-sonnet-4-5-20250929": {"max_tokens": 16384, "temperature": 0.0},
}

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        import anthropic
        assert "ANTHROPIC_API_KEY" in os.environ, "Please set the ANTHROPIC_API_KEY environment variable"
        CLIENT = anthropic.Anthropic()
    return CLIENT


def process(request, model, max_tokens, temperature):
    cache = get_cache("anthropic")
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

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": request['prompt']}]
    )

    return response.model_dump(mode="json")


def _extract(raw, temperature):
    if raw['stop_reason'] == "max_tokens":
        finish_reason = FINISH_LENGTH
    elif raw['stop_reason'] == "end_turn":
        finish_reason = FINISH_STOP
    else:
        logging.warning(f"Finish reason: {raw['stop_reason']}; {raw['content'][0]['text']}")
        return None

    return raw['content'][0]['text'], {
        "raw_response": raw,
        "model": raw['model'],
        "temperature": temperature,
        "reasoning_trace": None,
        "input_tokens": raw['usage']['input_tokens'],
        "output_tokens": raw['usage']['output_tokens'],
        "thinking_tokens": 0,
        "finish_reason": finish_reason
    }
