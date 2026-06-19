import os
from tools.cache import get_cache, cache_key
from tools.errors import FINISH_STOP, FINISH_LENGTH

MODELS = {
    "gpt-5.1": {"max_tokens": 32768, "temperature": 0.0},
}

CLIENT = None
def lazy_get_client():
    global CLIENT
        
    if CLIENT is None:
        from openai import OpenAI

        assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable"
        CLIENT = OpenAI()
    return CLIENT


def process(request, model, max_tokens, temperature):
    cache = get_cache("openai")
    key = cache_key(model, request)

    if key in cache:
        raw = cache[key]
    else:
        raw = _call(request, model, max_tokens)
        if raw is None:
            return None
        cache[key] = raw

    return _extract(raw)


def _call(request, model, max_tokens):
    import openai

    client = lazy_get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": request['prompt']}
            ],
            max_completion_tokens=max_tokens,
            reasoning_effort="none",
        )
    except (openai.BadRequestError, openai.APITimeoutError) as e:
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        raise e

    return response.model_dump(mode="json")


def _extract(raw):
    if raw['choices'][0]['finish_reason'] == "length":
        finish_reason = FINISH_LENGTH
    elif raw['choices'][0]['finish_reason'] == "stop":
        finish_reason = FINISH_STOP
    else:
        return None

    return raw['choices'][0]['message']['content'], {
        "raw_response": raw,
        "model": raw['model'],
        "temperature": None,
        "reasoning_trace": None,
        "input_tokens": raw['usage']['prompt_tokens'],
        "output_tokens": raw['usage']['completion_tokens'],
        "thinking_tokens": 0,
        "finish_reason": finish_reason
    }
