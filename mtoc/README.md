# mtoc: MT Output Collector

`mtoc` collects machine-translation outputs from model backends. The current WMT26 path focuses on offline/open-source LLMs through Hugging Face Transformers with Accelerate/device-map sharding. The code is structured so online/API providers can be wrapped later without changing input/output handling.

The clean entrypoint is `collect.py`. It reads JSONL rows like:

```json
{"doc_id":"00001-01","source_doc":"...","tgt_lang":"deu_Latn","instruction":"Translate ..."}
```

Required fields:

- `doc_id`: stable document or segment id
- `source_doc`: source text to translate
- `tgt_lang`: target language identifier copied to the output
- `instruction`: prompt/instruction text prepended to `source_doc`

Optional multimodal fields may exist in the input, but this text CLI currently rejects rows with non-empty `multimodal_instruction` or `multimodal_input_path`.

## Architecture

- `interfaces.py`: shared `TranslationRequest`, `TranslationResult`, `RuntimeModelConfig`, and a plain `BaseBackend` class.
- `offline.py`: offline backends for local model inference. The WMT26 registry defaults to Hugging Face Transformers; vLLM remains available for targeted experiments.
- `online.py`: adapter boundary for API/online providers. Existing provider functions can be wrapped if they accept a request dictionary and return either text or `(text, metadata)`.
- `collect.py`: CLI, input validation, resumable JSONL writing, failure logging, and GPU launch planning.
- `model_registry.wmt26.json`: default WMT26 offline model registry with only planned local systems.
- `run_manifest.json`: full model run list from the planning sheet, including offline, online, proprietary, and multi-node models.

## Install

For Transformers backends:

```bash
pip install -r requirements.txt
```

For optional vLLM experiments, install a vLLM build compatible with the node's CUDA/PyTorch stack:

```bash
pip install vllm
```

Some registered models are gated. Authenticate with Hugging Face before running them:

```bash
huggingface-cli login
```

## Download Models

Download all WMT26 local/offline model repos into the repo-level Hugging Face cache:

```bash
python download_models.py --cache-dir ../models/hf-hub --models all
```

The downloader reads `model_registry.wmt26.json`, enables Hugging Face fast-transfer settings, and stores snapshots under `../models/hf-hub`. To preview without downloading:

```bash
python download_models.py --dry-run
```

## Quick Checks

List registered models:

```bash
python collect.py list-models
```

Validate the sample input and selected models:

```bash
python collect.py validate --input ../tmp.jsonl --models qwen3_5_9b tower_9b
```

Print launch commands with GPU assignments:

```bash
python collect.py plan \
  --input ../tmp.jsonl \
  --models qwen3_5_9b ministral3_14b qwen3_6_27b gpt_oss_120b \
  --gpus 0,1,2,3,4,5,6,7 \
  --gnu-parallel
```

Run one small model on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python collect.py run \
  --input ../tmp.jsonl \
  --models qwen3_5_9b \
  --output-dir outputs \
  --log-dir logs
```

Run a multi-GPU HF/Accelerate model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python collect.py run \
  --input ../tmp.jsonl \
  --models qwen3_6_27b \
  --device-map auto \
  --output-dir outputs \
  --log-dir logs
```

## Outputs

Successful translations are written to:

```text
outputs/<model_key>/translations.jsonl
```

Failures are written to:

```text
logs/<model_key>/failures.jsonl
```

Each output row preserves the original input row and adds:

- `model_key`
- `model`
- `backend`
- `translation`
- `timestamp`
- `generation_params`

The runner resumes by default. It skips rows already present in the model's output file using `(doc_id, tgt_lang, model_key)`. Use `--no-resume` to rerun everything.

## Offline Backends

Offline backends implement the common interface in `interfaces.py`:

- `hf`: loads an instruction/chat model with `AutoModelForCausalLM`, `AutoTokenizer`, and Accelerate `device_map` support.
- `vllm`: optional experimental backend for models where first-run compile/capture overhead is acceptable.

Both backends receive `TranslationRequest` objects and return `TranslationResult` objects. The CLI is responsible for batching, resume checks, output rows, and failure rows.

## Online Backends

`online.py` contains `OnlineProviderBackend`, a small wrapper for future API-based implementations. It does not import the top-level provider code yet. Later, a colleague's provider can be consumed by wrapping a callable:

```python
backend = OnlineProviderBackend(model_config, provider_callable)
```

The callable receives a normalized request dictionary containing the original row plus `prompt`, `max_new_tokens`, `temperature`, and `top_p`.

## Model Registry

WMT26 model defaults live in `model_registry.wmt26.json`. Add or edit entries there for the actual WMT26 local/offline model list. Important fields:

- `hf_id`: Hugging Face model id
- `backend`: `hf` by default for WMT26; `vllm` only for targeted experiments
- `tensor_parallel_size`: number of GPUs to make visible for this model; HF uses them through `device_map=auto`
- `default_batch_size`
- `default_max_input_length`
- `default_max_new_tokens`
- `dtype`
- `trust_remote_code`
- `track`
- `run_scope`

The `plan` command uses `tensor_parallel_size` to print non-overlapping GPU waves. Commands inside one wave can run concurrently. Run later waves only after the previous wave finishes.

`run_manifest.json` is broader than `model_registry.wmt26.json`: it mirrors the planning sheet and includes proprietary, external, or multi-node models that `mtoc` should not try to launch as local GPU jobs yet. `model_registry.wmt26.json` should contain only WMT26 entries that `collect.py run` can execute locally with `hf` or `vllm`.

## Notes

- `collect.py run --models a b c` runs selected models sequentially in one process.
- For parallel collection, prefer `collect.py plan` and launch one process per model.
- `--backend registry` uses each model's registry default. For WMT26 this is `hf`; `--backend vllm` can be used for targeted experiments.
- `--temperature 0` uses greedy decoding for deterministic outputs.
