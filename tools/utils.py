import re
import os
import time
import ipdb
import glob
import logging
import threading
import traceback
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tools.providers import cohere, together_ai, openai, anthropic, google_translate, yandex_translate, mistral, gemini, microsoft_translator, deepl
from tools.errors import ERROR_UNSUPPORTED_LANGUAGE


_PROVIDERS = [cohere, together_ai, openai, anthropic, mistral, gemini, deepl, google_translate, yandex_translate, microsoft_translator]

MODELS = {}
for _p in _PROVIDERS:
    for _name, _params in _p.MODELS.items():
        MODELS[_name] = (_p.process, _params)


def _request_model(model_name, request):
    request['prompt'] = f"{request['instruction']}\n\n{request['segment']}"

    process, params = MODELS[model_name]
    answer = process(request, model=model_name, **params)

    if answer is None:
        print(f"Model {model_name} returned None for doc_id {request['doc_id']}")
        return None
    if answer == ERROR_UNSUPPORTED_LANGUAGE:
        raise ValueError(ERROR_UNSUPPORTED_LANGUAGE)

    answer, metadata = answer

    answer = answer.strip()
    return {
        'doc_id': request['doc_id'],
        'translation': answer,
        'translation_granularity': 'document-level',
        "metadata": metadata
    }


def _process_row(row, model_name, unsupported_languages, lock):
    # completely skip unsupported languages
    with lock:
        if row['tgt_lang'] in unsupported_languages:
            return None

    request = {
        'doc_id': row['doc_id'],
        'target_language': row['tgt_lang'],
        'segment': row['source_doc'],
        'instruction': row['instruction']
    }

    try:
        answer = _request_model(model_name, request)
    except Exception as e:
        if str(e) == ERROR_UNSUPPORTED_LANGUAGE:
            with lock:
                unsupported_languages.append(row['tgt_lang'])
            return None
        logging.error(f"Error processing {request['doc_id']} with {model_name}: {e}")
        logging.error(traceback.format_exc())
        answer = None

    if answer is None:
        answer = {
            'doc_id': request['doc_id'],
            'translation': "FAILED",
            'translation_granularity': None,
            'metadata': None
        }

    # mandatory fields for submission to OCELoT
    answer['dataset_id'] = "wmt26"
    answer['tgt_lang'] = row['tgt_lang']
    answer['hypothesis'] = answer.pop('translation')
    return answer


def collect_answers(blindset, model_name, num_workers):
    unsupported_languages = []
    lock = threading.Lock()
    rows = list(blindset.iterrows())
    results = [None] * len(rows)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_row, row, model_name, unsupported_languages, lock): i
            for i, (_, row) in enumerate(rows)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=model_name):
            results[futures[future]] = future.result()

    return [r for r in results if r is not None]