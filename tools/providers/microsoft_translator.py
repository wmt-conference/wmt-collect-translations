import os
import uuid
import ipdb
import requests
from tqdm import tqdm
from retrying import retry
from tools.errors import FINISH_STOP


def get_headers(MTAPI_SUBSCRIPTION_KEY):
    MTAPI_REGION = os.environ.get("MTAPI_REGION", "eastus")
    headers = {
        'Ocp-Apim-Subscription-Key': MTAPI_SUBSCRIPTION_KEY,
        'Ocp-Apim-Subscription-Region': MTAPI_REGION,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4()) }

    return headers

@retry(stop_max_attempt_number=5, retry_on_exception=lambda exception: isinstance(exception, ConnectionError))
def translate_with_microsoft_api(request, temperature=None, endpoint="https://api.cognitive.microsofttranslator.com/translate"):
    assert "MTAPI_SUBSCRIPTION_KEY" in os.environ, "Please set the MTAPI_SUBSCRIPTION_KEY environment variable."

    source_language = request.get('source_language')
    target_language = request['target_language']

    params = {
        'api-version': '2026-06-06',
    }

    input_entry = {'text': request['segment'], 'targets': [{'language': target_language}]}
    if source_language is not None:
        input_entry["language"] = source_language

    body = {'inputs': [input_entry]}
    http_request = requests.post(endpoint, params=params, headers=get_headers(os.environ["MTAPI_SUBSCRIPTION_KEY"]), json=body)
    response = http_request.json()

    assert len(response["value"][0]["translations"]) == 1, "More than one translation returned, this needs to be investigated."
    return response["value"][0]["translations"][0]['text'], {
        "raw_response": response,
        "model": None,
        "temperature": None,
        "reasoning_trace": None,
        "input_tokens": None,
        "output_tokens": None,
        "thinking_tokens": None,
        "finish_reason": FINISH_STOP
    }

def bulk_translate_with_microsoft(segments, source_pt1_iso, target_pt1_iso):
    assert "MTAPI_SUBSCRIPTION_KEY" in os.environ, "Please set the MTAPI_SUBSCRIPTION_KEY environment variable."

    translations = []
    for source_seg in tqdm(segments, "Translating with Microsoft API"):
        try:
            translation = translate_with_microsoft_api(source_seg, os.environ["MTAPI_SUBSCRIPTION_KEY"], src=source_pt1_iso, trg=target_pt1_iso)
        except Exception as e:
            print(e)
            return
        if len(translation) > 1:
            print("More than one translation returned, this needs to be investigated.")
            ipdb.set_trace()
        translations.append(translation[0])
        
    return translations, {"finish_reason": FINISH_STOP}
