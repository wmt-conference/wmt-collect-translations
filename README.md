# WMT Collecting Translations

This tool is used to collect translations from various providers and LLMs that may be used at WMT General MT 2025.
See the git tags for access to previous years.


# Usage

The tool is using 0-shot instruction following when translating with LLMs.

## Setting up secrets

You need to set one or multiple following secrets for the full utilization:

```
export MTAPI_SUBSCRIPTION_KEY=          # Microsoft Azure API key
export OPENAI_API_KEY=                  # Google credentials in json file
export DEEPL_PRO_AUTH_KEY=              # DeepL credentials
export YANDEX_APPLICATION_CREDENTIALS=  # Yandex API key
export TOGETHER_API_KEY=                # Together API key
export COHERE_API_KEY=                  # Cohere key
export OPENAI_API_KEY=                  # OpenAI Azure key
export MISTRAL_API_KEY=                 # Mistral API key
export GEMINI_API_KEY=                  # Gemini API key for Google AI Studio
export ANTHROPIC_API_KEY=               # Anthropic key for claude
```


## Running translations

```
python main.py --model='MODEL'
```

`MODEL` is the API model id (e.g. `command-a-plus-05-2026`); the owning provider and its default parameters are resolved from the per-provider `MODELS` configs in `tools/providers/`. Non-LLM translators are invoked by name (`DeepL`, `GoogleTranslate`, `YandexTranslate`, `MicrosoftTranslator`).

