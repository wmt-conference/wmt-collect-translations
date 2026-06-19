import re
import os
import time
import ipdb
import glob
import logging
import traceback
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from tools.providers import cohere, together_ai, openai, anthropic, google_translate, yandex_translate, mistral, gemini, microsoft_translator, deepl
from tools.errors import ERROR_UNSUPPORTED_LANGUAGE


_PROVIDERS = [cohere, together_ai, openai, anthropic, mistral, gemini, deepl, google_translate, yandex_translate, microsoft_translator]

MODELS = {}
for _p in _PROVIDERS:
    for _name, _params in _p.MODELS.items():
        MODELS[_name] = (_p.process, _params)


def _request_model(model_name, request):
    request['prompt'] = f"{request['prompt_instruction']}\n\n{request['segment']}"

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


def collect_answers(blindset, model_name):
    answers = []
    unsupported_languages = []
    for _, row in tqdm(blindset.iterrows(), total=len(blindset), desc=model_name):
        # completely skip unsupported languages
        if (row['src_lang'], row['tgt_lang']) in unsupported_languages:
            continue

        request = {
            'doc_id': row['doc_id'],
            'source_language': row['src_lang'],
            'target_language': row['tgt_lang'],
            'segment': row['src_text'],
            'prompt_instruction': row['prompt_instruction']
        }

        try:
            answer = _request_model(model_name, request)
        except Exception as e:
            if str(e) == ERROR_UNSUPPORTED_LANGUAGE:
                unsupported_languages.append((row['src_lang'], row['tgt_lang']))
                continue
            logging.error(f"Error processing {request['doc_id']} with {model_name}: {e}")
            logging.error(traceback.format_exc())
            answer = None

        if answer is not None:
            answers.append(answer)
        else:
            answers.append({
                'doc_id': request['doc_id'],
                'translation': "FAILED", # if everything fails, there is nothing we can do
                'translation_granularity': None,
            })
        # mandatory fields for submission to OCELoT
        answers[-1]['dataset_id'] = "wmttest2025"
        answers[-1]['tgt_lang'] = row['tgt_lang']
        answers[-1]['hypothesis'] = answers[-1].pop('translation')

    return answers