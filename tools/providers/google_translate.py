import os
import pandas as pd
from tools.errors import ERROR_UNSUPPORTED_LANGUAGE, FINISH_STOP

CLIENT = None
SUPPORTED_LANGUAGES = None
def lazy_get_client():
    global CLIENT, SUPPORTED_LANGUAGES
    
    if CLIENT is None:
        from google.cloud import translate_v2 as translate
        assert "GOOGLE_API_KEY" in os.environ, "Please set the GOOGLE_API_KEY environment variable"
        CLIENT = translate.Client()
    
        SUPPORTED_LANGUAGES = pd.DataFrame(CLIENT.get_languages())

    return CLIENT


def get_supported_languages(lang):
    lazy_get_client()
    if lang in SUPPORTED_LANGUAGES['language'].values:
        return lang
    elif lang.split('_')[0] in SUPPORTED_LANGUAGES['language'].values:
        return lang.split('_')[0]
    else:
        return ERROR_UNSUPPORTED_LANGUAGE
        

def translate_with_google_api(request, temperature=None):    
    goog_translate_client = lazy_get_client()

    target_language = get_supported_languages(request['target_language'])
    if target_language == ERROR_UNSUPPORTED_LANGUAGE:
        return ERROR_UNSUPPORTED_LANGUAGE

    result = goog_translate_client.translate(
                    request['segment'],
                    source_language=request['source_language'],
                    target_language=target_language,
                )

    return result.get('translatedText'), {
        "raw_response": result,
        "model": None,
        "temperature": None,
        "reasoning_trace": None,
        "input_tokens": None,
        "output_tokens": None,
        "thinking_tokens": None,
        "finish_reason": FINISH_STOP
    }
