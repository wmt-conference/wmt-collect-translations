import os
import requests
from tools.errors import ERROR_UNSUPPORTED_LANGUAGE, FINISH_STOP

ENDPOINT = "https://translate.api.cloud.yandex.net/translate/v2/translate"


def translate_with_yandex(request, temperature=None):
    assert 'YANDEX_API_KEY' in os.environ, 'Please set the environment variable YANDEX_API_KEY'
    assert 'YANDEX_FOLDER_ID' in os.environ, 'Please set the environment variable YANDEX_FOLDER_ID'

    source_language = request['source_language']
    target_language = request['target_language'].split("_")[0]  # Handle cases like 'en_US' to 'en'

    body = {
        "folderId": os.environ['YANDEX_FOLDER_ID'],
        "sourceLanguageCode": source_language,
        "targetLanguageCode": target_language,
        "texts": [request['segment']],
        "format": "PLAIN_TEXT",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}",
    }

    http_request = requests.post(ENDPOINT, json=body, headers=headers)
    response = http_request.json()

    if http_request.status_code != 200:
        message = response.get('message', '')
        if 'unsupported' in message.lower() and 'language' in message.lower():
            return ERROR_UNSUPPORTED_LANGUAGE
        raise RuntimeError(f"Yandex Translate API error: {http_request.status_code} - {message}")

    return response['translations'][0]['text'], {
        "raw_response": response,
        "model": None,
        "temperature": None,
        "reasoning_trace": None,
        "input_tokens": None,
        "output_tokens": None,
        "thinking_tokens": None,
        "finish_reason": FINISH_STOP
    }
