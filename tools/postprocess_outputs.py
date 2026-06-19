"""
Title: Script to postprocess Gemini Translations
Author: Sweta Agrawal
"""

import pandas as pd
import hashlib
import sys


def get_hash_genmt(row):
    hashid = f"{row['doc_id']}_{row['src_lang']}_{row['tgt_lang']}_{row['src_text']}_{row['prompt_instruction']}"
    hashid = hashlib.md5(hashid.encode("utf-8")).hexdigest()
    return f"{hashid}"


task = sys.argv[1]
if task == "genmt":
    blindset = pd.read_json(
        "data/genmt_blindset_wmt25.jsonl",
        lines=True,
    )
    blindset["key"] = blindset.apply(get_hash_genmt, axis=1)
    outputs = pd.read_json(
        "outputs_final/wmt-general_mt.jsonl",
        lines=True,
    )
else:
    print("Task not found")
    exit(0)


assert len(blindset) == len(outputs)

all_outs = []
count_max_tokens = 0
count_prohibit = 0
count_nan = 0
for x in outputs["response"].to_list():
    if x != x or "candidates" not in x:
        count_nan += 1
        all_outs.append(None)
        continue
    x = x["candidates"][0]
    if x["finishReason"] == "MAX_TOKENS":
        count_max_tokens += 1
        all_outs.append(None)
        continue
    elif x["finishReason"] == "PROHIBITED_CONTENT":
        count_prohibit += 1
        all_outs.append(None)
        continue
    else:
        if "content" in x and "parts" in x["content"]:
            all_outs.append(x["content"]["parts"][0]["text"])
        else:
            all_outs.append(None)

print(count_max_tokens, count_prohibit, count_nan, len(outputs))

outputs["response_text"] = all_outs
all_keys_out = outputs["key"].to_list()
all_keys_blindset = blindset["key"].to_list()

merged_df = pd.merge(
    blindset, outputs, on="key", how="inner", suffixes=("_test", "_out")
)

assert len(merged_df) == len(outputs)
