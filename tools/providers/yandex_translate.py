import os
from tools.errors import ERROR_UNSUPPORTED_LANGUAGE, FINISH_STOP

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from yandex_translate import YandexTranslate
        assert 'YANDEX_APPLICATION_CREDENTIALS' in os.environ, 'Please set the environment variable YANDEX_APPLICATION_CREDENTIALS'
        CLIENT = YandexTranslate(os.environ['YANDEX_APPLICATION_CREDENTIALS'])
    return CLIENT


def translate_with_yandex(request, temperature=None):
    client = lazy_get_client()
    
    source_language = request['source_language']
    target_language = request['target_language'].split("_")[0]  # Handle cases like 'en_US' to 'en'
    
    try:
        result = client.translate(request['segment'], f'{source_language}-{target_language}')
    except Exception as err:
        if str(err) == 'ERR_TEXT_TOO_LONG':
            return None
        if str(err) == 'ERR_LANG_NOT_SUPPORTED':
            return ERROR_UNSUPPORTED_LANGUAGE
        raise err

    assert result.get('code') == 200, f"Yandex Translate API error: {result.get('code')} - {result.get('text')}"

    return result.get('text')[0], {
        "raw_response": result,
        "model": None,
        "temperature": None,
        "reasoning_trace": None,
        "input_tokens": None,
        "output_tokens": None,
        "thinking_tokens": None,
        "finish_reason": FINISH_STOP
    }
