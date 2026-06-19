import os
from tools.cache import get_cache, cache_key
from tools.errors import ERROR_UNSUPPORTED_LANGUAGE, FINISH_STOP

MODELS = {
    "GoogleTranslate": {},
}

# Maps wmt26_genmt_blindset.jsonl tgt_lang tags to the NMT language codes Google Translate uses.
# None means the language is missing from Google Translate's supported list.
GOOGLE_LANGUAGE_CODES = {
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
    "en": "en",
    "en_US": "en",
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
    "lij_Latn": "lij",    # Ligurian
    "lld_Latn": None,     # Ladin
    "mni_Beng": "mni-Mtei",
    "mni_Latn": "mni-Mtei",
    "mni_Mtei": "mni-Mtei",
    "pl_PL": "pl",
    "ru": "ru",
    "ru_RU": "ru",
    "rus_Cyrl": "ru",
    "sme_Latn": None,     # Northern Sami
    "tha_Thai": "th",
    "ukr_Cyrl": "uk",
    "vie_Latn": "vi",
    "zh_CN": "zh-CN",
    "zho_Hans": "zh-CN",
    "zho_Hant_TW": "zh-TW",
}

CLIENT = None
def lazy_get_client():
    global CLIENT

    if CLIENT is None:
        from google.cloud import translate_v2 as translate
        assert "GOOGLE_API_KEY" in os.environ, "Please set the GOOGLE_API_KEY environment variable"
        CLIENT = translate.Client()

    return CLIENT


def get_supported_languages(lang):
    target = GOOGLE_LANGUAGE_CODES.get(lang)
    if target is None:
        return ERROR_UNSUPPORTED_LANGUAGE
    return target
        

def process(request, model=None):
    goog_translate_client = lazy_get_client()

    target_language = get_supported_languages(request['target_language'])
    if target_language == ERROR_UNSUPPORTED_LANGUAGE:
        print(f"Unsupported language: {request['target_language']}")
        return ERROR_UNSUPPORTED_LANGUAGE

    cache = get_cache("google_translate")
    key = cache_key(model, request)

    if key in cache:
        raw = cache[key]
    else:
        raw = goog_translate_client.translate(
            request['segment'],
            target_language=target_language,
        )
        cache[key] = raw

    return raw.get('translatedText'), {
        "raw_response": raw,
        "model": None,
        "extra": None,
        "reasoning_trace": None,
        "input_tokens": None,
        "output_tokens": None,
        "thinking_tokens": None,
        "finish_reason": FINISH_STOP
    }
