# WMT Collecting Translations

This tool is used to collect translations from various providers and LLMs that may be used at WMT General MT 2025.
See the git tags for access to previous years.


# Usage

The tool is using 0-shot instruction following when translating with LLMs.

## Setting up secrets

Copy `secrets.env.example` to `secrets.env` (this file is git-ignored) and fill in the
keys for the providers you want to use. Keys you leave empty are ignored, so you only
need the ones relevant to the models you run. The file is loaded automatically by `main.py`.

```
cp secrets.env.example secrets.env
```


## Running translations

```
python main.py --model='MODEL'
```

`MODEL` is the API model id (e.g. `command-a-plus-05-2026`); the owning provider and its default parameters are resolved from the per-provider `MODELS` configs in `tools/providers/`. Non-LLM translators are invoked by name (`DeepL`, `GoogleTranslate`, `YandexTranslate`, `MicrosoftTranslator`).

