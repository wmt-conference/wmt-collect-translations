import re
import os
import time
import ipdb
import glob
import logging
import traceback
import hashlib
import pandas as pd
import diskcache as dc
from tqdm import tqdm
from collections import defaultdict

from tools.providers.cohere import process_with_command_A, process_with_command_R7B, process_with_aya_expanse_32B, process_with_aya_expanse_8B
from tools.providers.together_ai import process_with_deepseek_v3, process_qwen3_235b, process_with_llama_4_maverick, process_with_llama_4_scout, process_with_mistral_7b, process_qwen25_7b, process_with_llama_3_1_8b
from tools.providers.openai import process_with_openai_gpt4_1
from tools.providers.anthropic import process_with_claude_4
from tools.providers.google_translate import translate_with_google_api
from tools.providers.yandex_translate import translate_with_yandex
from tools.providers.mistral import process_with_mistral_medium
from tools.providers.gemini import process_with_gemini_2_5_pro, process_with_gemma_3_12b, process_with_gemma_3_27b
from tools.providers.microsoft_translator import translate_with_microsoft_api
from tools.providers.deepl import translate_with_deepl
from tools.errors import ERROR_UNSUPPORTED_LANGUAGE


SYSTEMS = {
    'CommandA': process_with_command_A,
    'CommandR7B': process_with_command_R7B,
    'AyaExpanse-32B': process_with_aya_expanse_32B,
    'AyaExpanse-8B': process_with_aya_expanse_8B,
    'DeepSeek-V3': process_with_deepseek_v3,
    'Qwen3-235B': process_qwen3_235b,
    'Qwen2.5-7B': process_qwen25_7b,
    'Llama-4-Maverick': process_with_llama_4_maverick,
    'Llama-4-Scout': process_with_llama_4_scout,
    'Llama-3.1-8B': process_with_llama_3_1_8b,
    'Mistral-7B': process_with_mistral_7b,
    'GPT-4.1': process_with_openai_gpt4_1,
    'Claude-4': process_with_claude_4,
    "Mistral-Medium": process_with_mistral_medium,
    'YandexTranslate': translate_with_yandex,
    'GoogleTranslate': translate_with_google_api,
    'Gemini-2.5-Pro': process_with_gemini_2_5_pro,
    'Gemma-3-12B': process_with_gemma_3_12b,
    'Gemma-3-27B': process_with_gemma_3_27b,
    'DeepL': translate_with_deepl,
    'MicrosoftTranslator': translate_with_microsoft_api,
}


def _request_system(system_name, request):
    request['prompt'] = f"{request['prompt_instruction']}\n\n{request['segment']}"

    answer = SYSTEMS[system_name](request)

    if answer is None:
        print(f"System {system_name} returned None for doc_id {request['doc_id']}")
        return None
    if answer == ERROR_UNSUPPORTED_LANGUAGE:
        raise ValueError(ERROR_UNSUPPORTED_LANGUAGE)

    answer, tokens = answer

    answer = answer.strip()
    return {
        'doc_id': request['doc_id'],
        'translation': answer,
        'translation_granularity': 'document-level',
        "tokens": tokens
    }


# TODO: WMT25 - track temperature and reasoning traces in metadata

def collect_answers(blindset, system_name):
    cache = dc.Cache(f'cache/{system_name}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')

    answers = []
    unsupported_languages = []
    for _, row in tqdm(blindset.iterrows(), total=len(blindset), desc=system_name):
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
        # create hash merging all the information in the request
        hashid = f"{request['doc_id']}_{request['source_language']}_{request['target_language']}_{request['segment']}_{request['prompt_instruction']}"
        hashid = hashlib.md5(hashid.encode('utf-8')).hexdigest()

        # None represent problem in the translation that was originally skipped
        if hashid not in cache or cache[hashid] is None:
            try:
                cache[hashid] = _request_system(system_name, request)
            except Exception as e:
                if str(e) == ERROR_UNSUPPORTED_LANGUAGE:
                    unsupported_languages.append((row['src_lang'], row['tgt_lang']))
                    continue
                logging.error(f"Error processing {request['doc_id']} with {system_name}: {e}")
                logging.error(traceback.format_exc())
                cache[hashid] = None

        if cache[hashid] is not None:
            answers.append(cache[hashid])
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