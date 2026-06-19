import os
import copy
import logging
from tools.cache import get_cache, cache_key
from tools.errors import FINISH_STOP, FINISH_LENGTH

MODELS = {
    "command-a-plus-05-2026": {"max_tokens": 8192, "temperature": 0.0},
    "command-r7b-12-2024": {"max_tokens": 4096, "temperature": 0.0},
    "c4ai-aya-expanse-32b": {"max_tokens": 4096, "temperature": 0.0},
}

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from cohere import ClientV2
        assert "COHERE_API_KEY" in os.environ, "Please set the COHERE_API_KEY environment variable"
        CLIENT = ClientV2(api_key=os.environ.get("COHERE_API_KEY"))
    return CLIENT


def process(request, model, max_tokens, temperature):
    cache = get_cache("cohere")
    key = cache_key(model, request)

    if key in cache:
        raw = cache[key]
    else:
        raw = _call(request, model, max_tokens, temperature)
        if raw is None:
            return None
        cache[key] = raw

    return _extract(raw, model, temperature)


def _call(request, model, max_tokens, temperature):
    import cohere

    co = lazy_get_client()
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": request['prompt']}]
    }]

    try:
        response = co.chat(
            model=model,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens,
        )
    except (cohere.errors.bad_request_error.BadRequestError, cohere.errors.unprocessable_entity_error.UnprocessableEntityError) as err:
        if 'too many tokens' in err.body['message']:
            return None
        if "No valid response generated" in err.body['message']:
            return None
        raise err

    return response.model_dump(mode="json")


def _extract(raw, model, temperature):
    if raw['finish_reason'] == 'MAX_TOKENS':
        finish_reason = FINISH_LENGTH
    elif raw['finish_reason'] == 'COMPLETE':
        finish_reason = FINISH_STOP
    else:
        logging.warning(f"Finish reason: {raw['finish_reason']}")
        return None

    return raw['message']['content'][0]['text'], {
        "raw_response": raw,
        "model": model,
        "temperature": temperature,
        "reasoning_trace": None,
        "input_tokens": raw['usage']['billed_units']['input_tokens'],
        "output_tokens": raw['usage']['billed_units']['output_tokens'],
        "thinking_tokens": 0,
        "finish_reason": finish_reason
    }
