"""
    Title: Machine Translation API command line tool for translating WMT testsets
    Author: Tom Kocmi
"""

import os
import ipdb
import urllib.request
import pandas as pd
from absl import flags, app
from tools.utils import collect_answers, MODELS
 

flags.DEFINE_enum('model', 'command-a-plus-05-2026', list(MODELS.keys()), 'Define the model to use for translation')
flags.DEFINE_bool('parallel', False, 'Run in parallel mode (default: False)')

FLAGS = flags.FLAGS

def main(args):
    if not os.path.exists("wmt26_genmt_blindset.jsonl"):
        print("Blindset not found, downloading from WMT website")
        urllib.request.urlretrieve("https://www2.statmt.org/wmt26/assets/wmt26_genmt_blindset.jsonl", "wmt26_genmt_blindset.jsonl")
    blindset = pd.read_json("wmt26_genmt_blindset.jsonl", lines=True)
    if FLAGS.parallel:
        # avoid clashes by shuffling samples
        blindset = blindset.sample(frac=1, random_state=42).reset_index(drop=True)

    answers = collect_answers(blindset, FLAGS.model)
    df = pd.DataFrame(answers)

    # for each tgt_lang, count how many "FAILED" there are and if more than 25% are FAILED, remove that tgt_lang
    for tgt_lang in df['tgt_lang'].unique():
        num_none = df[df['tgt_lang'] == tgt_lang]['hypothesis'].str.contains("FAILED", na=False).sum()
        if num_none > 0.25 * len(df[df['tgt_lang'] == tgt_lang]):
            df = df[df['tgt_lang'] != tgt_lang]

    if not FLAGS.parallel:
        os.makedirs("wmt_translations", exist_ok=True)
        df.to_json(f"wmt_translations/{FLAGS.model.replace('/', '_')}.jsonl", orient='records', lines=True, force_ascii=False)
    else:
        print("Running in parallel mode, not saving results to disk as the data are shuffled.")

    mt_num_none = df[df['hypothesis'].str.contains("FAILED", na=False)]['hypothesis'].count()
    print(f"Number of untranslated answers in MT: {mt_num_none}")


if __name__ == '__main__':
    app.run(main)
