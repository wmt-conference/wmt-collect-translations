import os
from tools.cache import get_cache, cache_key
from tools.errors import ERROR_UNSUPPORTED_LANGUAGE, FINISH_STOP

MODELS = {
    "DeepL": {},
}

CLIENT = None
SUPPORTED_LANGUAGES = None
def lazy_get_client():
    global CLIENT, SUPPORTED_LANGUAGES
    
    if CLIENT is None:
        import deepl
        assert "DEEPL_PRO_AUTH_KEY" in os.environ, "Please set the DEEPL_PRO_AUTH_KEY environment variable"
        CLIENT = deepl.DeepLClient(os.environ['DEEPL_PRO_AUTH_KEY'])
        SUPPORTED_LANGUAGES = {lang.code.lower() for lang in CLIENT.get_target_languages()}
    return CLIENT


def process(request, model=None):
    client = lazy_get_client()

    target_language = request['target_language'].split('_')[0]
    if target_language not in SUPPORTED_LANGUAGES:
        return ERROR_UNSUPPORTED_LANGUAGE

    cache = get_cache("deepl")
    key = cache_key(model, request)

    if key in cache:
        raw = cache[key]
    else:
        result = client.translate_text(
            request['segment'],
            source_lang=request['source_language'],
            target_lang=target_language,
        )
        raw = {"text": result.text, "detected_source_lang": result.detected_source_lang}
        cache[key] = raw

    return raw['text'], {
        "raw_response": raw,
        "model": None,
        "temperature": None,
        "reasoning_trace": None,
        "input_tokens": None,
        "output_tokens": None,
        "thinking_tokens": None,
        "finish_reason": FINISH_STOP
    }
