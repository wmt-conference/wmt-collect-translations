# Decoding Strategies for LLM Machine Translation — Findings

A controlled study of how decoding strategy affects translation quality for
instruction-tuned LLMs, measured with reference-free CometKIWI22 (wmt22-cometkiwi-da).

## Setup

- **Test set:** WMT25 general MT, subsampled to **N=300** segments (150 `en→ru_RU` + 150 `en→zh_CN`).
- **Models (3 families, bf16, tp=1):** `qwen3_5_9b` (Qwen), `glm4_9b` (GLM), `ministral3_14b` (Mistral).
- **Decoding (uniform `max_new_tokens=512` across all variants — fair comparison):**
  - greedy
  - sampling temperature sweep `t ∈ {0.3, 0.5, 0.7, 0.9, 1.1}` (top_p=0.95)
  - top_p sweep `{0.80, 0.90, 1.00}` (t=0.7)
  - beam size `{2, 4}`, beam length-penalty `{0.6, 1.4}` (beam=4)
- **Metric:** CometKIWI22, mean over segments (higher is better).

## Key findings

### Q1 — Beam search vs sampling: sampling/greedy win decisively

Aggregate mean CometKIWI22 (across the models holding each variant):

| Method | Mean | n |
|---|---|---|
| top_p=0.80 (t0.7) | **0.766** | 900 |
| sample t=0.5 | 0.766 | 900 |
| sample t=0.3 | 0.765 | 900 |
| **greedy** | **0.765** | 900 |
| sample t=0.7 | 0.760 | 900 |
| sample t=0.9 | 0.738 | 900 |
| **beam4** | **0.695** | 600 |
| **beam2** | **0.694** | 600 |
| sample t=1.1 | 0.576 | 900 |

Per-model (greedy vs beam4), the gap is robust across families:

| Model | greedy | beam4 | Δ (beam − greedy) |
|---|---|---|---|
| qwen3_5_9b | 0.765 | 0.695 | −0.070 |
| glm4_9b | 0.764 | 0.641 | −0.123 |

**Beam search is consistently ~0.07–0.12 CometKIWI worse than greedy / low-temperature
sampling for LLM MT.** Wider beams do not help (beam2 ≈ beam4), and beam length-penalty
(0.6 / 1.4) does not recover the gap (glm beam ≈ 0.64 across all penalties). This matches
the broader finding that beam search degrades quality for instruction-tuned LLM decoders.

### Q2 — Parameter sensitivity

- **Temperature:** flat and near-best from greedy through t=0.7; mild drop at t=0.9;
  **collapses at t=1.1** (0.576 aggregate; e.g. en→ru qwen 0.80→0.56). Keep **t ≤ 0.7**.
- **top_p:** **0.80 is best** (0.766) and degrades gently toward 1.00 (0.751). Slightly
  lower top_p helps.
- **Beam size:** beam2 ≈ beam4 — no benefit from wider beams.
- **Language pair:** `en→ru_RU` scores ~0.06 higher than `en→zh_CN` (Chinese is harder),
  but the *shape* of every curve is identical across pairs.

### Practical recommendation

For LLM MT, use **greedy** (deterministic, reproducible, ties for best) or **low-temperature
sampling** (t≤0.5, top_p≈0.80). **Avoid beam search** and **avoid t ≥ 1.0**.

> Note on the WMT26 blindset collection: it used `t=0.7, top_p=0.95`, which is close to
> optimal but ~0.005–0.006 below the t=0.5 / top_p=0.80 / greedy sweet spot. Not a
> meaningful regression, but greedy or t=0.5 would be a marginally better default.

## Plots

`outputs/decoding-study/plots/`: `temperature_sweep.png`, `topp_sweep.png`,
`beam_sweep.png`, `method_comparison.png` (per-language-pair lines).

## Caveats

- **ministral3_14b beam excluded:** the beam path re-encodes the rendered chat string,
  which Mistral's `tokenizer_mode="mistral"` mangles (literal `[INST]`) → empty outputs.
  Its sampling/top_p data is included; its beam data is not. (Fixable in `_beam_search`.)
- **qwen length-penalty (lp06/lp14)** points are glm-only at the time of writing
  (qwen's were still running — `lp=1.4` favors long sequences and is very slow).
- Beam search is **expensive**: ~6–22 s/row vs ~1 s/row for sampling. See the runtime notes.

## mtoc changes required (surgical)

The study exposed real issues in the shared-task vLLM adapter; fixes are additive and opt-in:

- `enable_prefix_caching=True` in `VllmAdapter.load()` — vLLM's `beam_search()` re-submits
  the full growing sequence each step; prefix caching makes it reuse the KV prefix
  (~O(T²)→~O(T)). Output-preserving (exact), benefits all paths.
- `concurrency_limit` (env `BEAM_CONCURRENCY_LIMIT`, default 64) on the `beam_search()` call —
  unbounded beam over 300 rows × beam_width 4 = 1200 live sequences thrashes the KV cache
  and effectively stalls (observed: 11 h, 0 output). Sub-batching keeps memory bounded.
- `quantization` registry field (opt-in, default off) — supports FP8 in the fast registry.

Pre-existing (not from this work): `mtoc/collect.py` uses `Any` in two annotations without
importing it from `typing`; harmless at runtime due to `from __future__ import annotations`.
