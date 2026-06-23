import os
import logging
from tools.cache import get_cache, cache_key
from tools.errors import FINISH_STOP, FINISH_LENGTH

MODELS = {
    "gemini-3.1-pro-preview": {"extra": {"max_output_tokens": 65536, "thinking_config": {"thinking_level": "medium"}}},
}

CLIENT = None
def lazy_get_client():
    global CLIENT

    if CLIENT is None:
        from google import genai
        assert "GEMINI_API_KEY" in os.environ, "Please set the GEMINI_API_KEY environment variable"
        CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return CLIENT


def process(request, model, extra=None):
    extra = extra or {}
    cache = get_cache("gemini")
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
    client = lazy_get_client()
    from google.genai import types

    config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        **extra,
        safety_settings=[
            types.SafetySetting(category=category, threshold=types.HarmBlockThreshold.BLOCK_NONE)
            for category in [
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        ],
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=request["prompt"],
            config=config
        )
    except Exception as e:
        logging.warning(f"Skipping: {e}")
        return None

    if response.candidates is None:
        return None

    return response.model_dump(mode="json")


def _extract(raw, extra):
    if raw['candidates'][0]['finish_reason'] == "MAX_TOKENS":
        finish_reason = FINISH_LENGTH
    elif raw['candidates'][0]['finish_reason'] == "STOP":
        finish_reason = FINISH_STOP
    else:
        logging.warning(f"Finish reason: {raw['candidates'][0]['finish_reason']}")
        return None

    input_tokens = raw['usage_metadata']['prompt_token_count']
    candidate_tokens = raw['usage_metadata']['candidates_token_count']
    thinking_tokens = raw['usage_metadata']['thoughts_token_count']

    if candidate_tokens is None:
        # gemma has only total prompt token count which equals to input tokens
        candidate_tokens = 0

    if thinking_tokens is None:
        thinking_tokens = 0

    text = "".join(part['text'] for part in raw['candidates'][0]['content']['parts'] if part.get('text'))

    return text, {"raw_response": raw,
                  "model": raw['model_version'],
                  "extra": extra,
                  "reasoning_trace": None,
                  "input_tokens": input_tokens,
                  "output_tokens": candidate_tokens + thinking_tokens,
                  "thinking_tokens": thinking_tokens,
                  "finish_reason": finish_reason}
