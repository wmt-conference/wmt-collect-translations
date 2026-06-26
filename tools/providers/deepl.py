import os
from tools.cache import get_cache, cache_key
from tools.errors import ERROR_UNSUPPORTED_LANGUAGE, FINISH_STOP

MODELS = {
    "deepl-nextgen": {"model_type": "quality_optimized"},
    "deepl-latency": {"model_type": "latency_optimized"},
}

# Maps wmt26_genmt_blindset.jsonl tgt_lang tags to the language codes DeepL uses.
# None means the language is missing from DeepL's supported list.
DEEPL_LANGUAGE_CODES = {
    "aeb": "ar",          # Tunisian Arabic
    "ar_AR": "ar",
    "arz": "ar",          # Egyptian Arabic
    "arz_Arab": "ar",
    "bel_Cyrl": "be",
    "ces_Latn": "cs",
    "cs": "cs",
    "cs_CZ": "cs",
    "de_AT": "de",
    "de_CH": "de",
    "de_DE": "de",
    "de_IT": "de",
    "deu_Latn": "de",
    "ekk_Latn": "et",     # Standard Estonian
    "en": "en-us",
    "en_US": "en-us",
    "es_ES": "es",
    "et_EE": "et",
    "fo": None,           # Faroese
    "hin_Deva": "hi",
    "hr": "hr",
    "hye_Armn": "hy",
    "ind_Latn": "id",
    "is": "is",
    "isl_Latn": "is",
    "jpn_Jpan": "ja",
    "kaz_Cyrl": "kk",
    "ko_KR": "ko",
    "kor_Hang": "ko",
    "lij_Latn": None,     # Ligurian
    "lld_Latn": None,     # Ladin
    "mni_Beng": None,     # Manipuri
    "mni_Latn": None,
    "mni_Mtei": None,
    "pl_PL": "pl",
    "ru": "ru",
    "ru_RU": "ru",
    "rus_Cyrl": "ru",
    "sme_Latn": None,     # Northern Sami
    "tha_Thai": "th",
    "ukr_Cyrl": "uk",
    "vie_Latn": "vi",
    "zh_CN": "zh-hans",
    "zho_Hans": "zh-hans",
    "zho_Hant_TW": "zh-hant",
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


def process(request, model=None, model_type=None):
    client = lazy_get_client()

    target_language = DEEPL_LANGUAGE_CODES.get(request['target_language'])
    if target_language is None or target_language not in SUPPORTED_LANGUAGES:
        return ERROR_UNSUPPORTED_LANGUAGE

    cache = get_cache("deepl")
    key = cache_key(model, request)

    if key in cache:
        raw = cache[key]
    else:
        result = client.translate_text(
            request['segment'],
            target_lang=target_language,
            model_type=model_type,
        )
        raw = {"text": result.text, "detected_source_lang": result.detected_source_lang, "model_type_used": result.model_type_used}
        cache[key] = raw

    return raw['text'], {
        "raw_response": raw,
        "model": None,
        "extra": None,
        "reasoning_trace": None,
        "input_tokens": None,
        "output_tokens": None,
        "thinking_tokens": None,
        "finish_reason": FINISH_STOP
    }
