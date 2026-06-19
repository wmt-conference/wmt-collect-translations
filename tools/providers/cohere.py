import os
import copy
import logging
from tools.cache import get_cache, cache_key
from tools.errors import FINISH_STOP, FINISH_LENGTH

MODELS = {
    # https://huggingface.co/CohereLabs/command-a-plus-05-2026-w4a4
    # "p" is Cohere's name for top_p
    "command-a-plus-05-2026": {"max_tokens": 64000, "extra": {"temperature": 0.9, "p": 0.95}},
    "tiny-aya-global": {"max_tokens": 8096, "extra": {}},
}

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from cohere import ClientV2
        assert "COHERE_API_KEY" in os.environ, "Please set the COHERE_API_KEY environment variable"
        CLIENT = ClientV2(api_key=os.environ.get("COHERE_API_KEY"))
    return CLIENT


def process(request, model, max_tokens, extra=None):
    extra = extra or {}
    cache = get_cache("cohere")
    key = cache_key(model, request)

    if key in cache:
        raw, extra = cache[key]["raw"], cache[key]["extra"]
    else:
        raw = _call(request, model, max_tokens, extra)
        if raw is None:
            return None
        cache[key] = {"raw": raw, "extra": extra}

    return _extract(raw, model, extra)


def _call(request, model, max_tokens, extra):
    import cohere

    co = lazy_get_client()
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": request['prompt']}]
    }]

    try:
        response = co.chat(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **extra,
        )
    except (cohere.errors.bad_request_error.BadRequestError, cohere.errors.unprocessable_entity_error.UnprocessableEntityError) as err:
        if 'too many tokens' in err.body['message']:
            return None
        if "No valid response generated" in err.body['message']:
            return None
        raise err

    return response.model_dump(mode="json")


def _extract(raw, model, extra):
    if raw['finish_reason'] == 'MAX_TOKENS':
        finish_reason = FINISH_LENGTH
    elif raw['finish_reason'] == 'COMPLETE':
        finish_reason = FINISH_STOP
    else:
        logging.warning(f"Finish reason: {raw['finish_reason']}")
        return None

    text = "".join(part['text'] for part in raw['message']['content'] if part['type'] == 'text')
    thinking = "".join(part['thinking'] for part in raw['message']['content'] if part['type'] == 'thinking')

    return text, {
        "raw_response": raw,
        "model": model,
        "extra": extra,
        "reasoning_trace": thinking or None,
        "input_tokens": raw['usage']['billed_units']['input_tokens'],
        "output_tokens": raw['usage']['billed_units']['output_tokens'],
        "thinking_tokens": raw['usage']['tokens'].get('reasoning_tokens', 0) or 0,
        "finish_reason": finish_reason
    }
