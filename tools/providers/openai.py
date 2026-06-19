import os
from tools.cache import get_cache, cache_key
from tools.errors import FINISH_STOP, FINISH_LENGTH

MODELS = {
    # https://developers.openai.com/api/docs/models/gpt-5.5
    "gpt-5.5-2026-04-23": {"extra": {"max_completion_tokens": 32768, "reasoning_effort": "medium", "verbosity": "medium"}},
}

CLIENT = None
def lazy_get_client():
    global CLIENT
        
    if CLIENT is None:
        from openai import OpenAI

        assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable"
        CLIENT = OpenAI()
    return CLIENT


def process(request, model, extra=None):
    extra = extra or {}
    cache = get_cache("openai")
    key = cache_key(model, request)

    if key in cache:
        raw, extra = cache[key]["raw"], cache[key]["extra"]
    else:
        raw = _call(request, model, extra)
        if raw is None:
            return None
        cache[key] = {"raw": raw, "extra": extra}

    return _extract(raw, extra)


def _call(request, model, extra):
    import openai

    client = lazy_get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": request['prompt']}
            ],
            **extra,
        )
    except (openai.BadRequestError, openai.APITimeoutError) as e:
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        raise e

    return response.model_dump(mode="json")


def _extract(raw, extra):
    if raw['choices'][0]['finish_reason'] == "length":
        finish_reason = FINISH_LENGTH
    elif raw['choices'][0]['finish_reason'] == "stop":
        finish_reason = FINISH_STOP
    else:
        return None

    return raw['choices'][0]['message']['content'], {
        "raw_response": raw,
        "model": raw['model'],
        "extra": extra,
        "reasoning_trace": None,
        "input_tokens": raw['usage']['prompt_tokens'],
        "output_tokens": raw['usage']['completion_tokens'],
        "thinking_tokens": raw['usage']['completion_tokens_details'].get('reasoning_tokens', 0) or 0,
        "finish_reason": finish_reason
    }
