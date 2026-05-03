# Method: differential competence-map pipeline

The pipeline produces **per-source per-element importance maps** suitable
for feeding into `omnimergekit.py --fisher`. The key idea: each source's
map is computed only on the docs **that source uniquely solved** vs the
other sources. The map answers the question "which parameters does THIS
source need to do the things ONLY it can do?"

## Why differential?

A naive Fisher signal collected on all task docs gives you a "what does
this source compute" map — useful, but it dilutes the merge: the
high-importance parameters are mostly things every source has, so
Fisher-weighted averaging looks a lot like uniform averaging.

When you restrict to docs source-i uniquely solved (i.e. doc passed for
source-i, failed for all others), the resulting map highlights parameters
that *distinguish* source-i. Merging with these maps preserves the unique
capability of each source rather than averaging it away.

This is why merges produced with this pipeline beat plain DARE-TIES on
multi-task benchmarks (HE, MBPP, LCB) for the same source set, but tie or
lose on single-task benchmarks where Fisher's diluting bias is actually
helpful.

## Pipeline

```
                 ┌──────────────────────────────────┐
                 │ For each (source, task):         │
                 │  1. lm_eval on source            │
                 │  2. read samples_*.jsonl         │
                 │  3. find docs that source solved │
                 │     AND no other source solved   │
                 └────────┬─────────────────────────┘
                          │
                          ▼
              ┌──────────────────────────────────────┐
              │ competence_extract.py                │
              │  - load source HF model              │
              │  - for each kept doc:                │
              │      forward → loss → backward       │
              │      accumulate (|grad|, grad², |W·grad|) │
              │      to per-param CPU fp32 buffers   │
              │  - export safetensors with all 3 signals │
              └────────┬─────────────────────────────┘
                       │ <source>__<task>.safetensors
                       ▼
              ┌──────────────────────────────────────┐
              │ competence_combine.py                │
              │  - per source: weighted-sum across   │
              │    tasks, weight = task pass-rate    │
              │  - choose ONE signal (default        │
              │    weight_taylor)                    │
              │  - normalize across sources          │
              │    per-element                       │
              └────────┬─────────────────────────────┘
                       │ combined/<source>.safetensors
                       ▼
              ┌──────────────────────────────────────┐
              │ omnimergekit.py --fisher ...         │
              └──────────────────────────────────────┘
```

## `competence_extract.py` — what to know

**Three signals are always exported.** Per parameter tensor:

| Suffix | Meaning |
|--------|---------|
| (no suffix) | `mean(|grad|)` — pure gradient magnitude. Robust, fast. |
| `.grad_sq` | `mean(grad²)` — Fisher information diagonal. The classical signal. |
| `.weight_taylor` | `mean(|W · grad|)` — first-order Taylor importance. Reflects pruning utility (low values prune cleanly). **Default for downstream `competence_combine.py`**. |

You can pick any one in `competence_combine.py --signal {grad_l1,grad_sq,weight_taylor}`.

### Memory: chunked gradient accumulation

For long-context tasks (AIME reasoning traces, LCB problems with full
test harness), a single forward+backward at 4-8k seq doesn't fit on a
24 GB GPU for a 4B model — the lm_head logits over a 250k vocab dominate
peak VRAM.

The fix: `--chunk-len N`. Splits each sample into windows of N tokens,
runs forward+backward per chunk, **flushes `.grad` to CPU between chunks**
(otherwise the bf16 grad buffer pins ~8 GB while the next chunk's
activations need it). Sample-level counter only increments at sample
boundary, so `competence_combine.py` divides by the right denominator.

Approximation: each chunk is processed as an independent sequence (no
cross-chunk attention or recurrent state). For Fisher importance this is
fine — we want per-param gradient magnitude, not exact training-style
gradients. For training itself, this approximation would be wrong.

### Filters

- `--keep-doc-ids "0,1,15,19,..."` — restrict to a precomputed
  differential set. Combine with the pass filter (`pass_value=1.0`)
  via the task preset.
- `--task {he,mbpp,lcb,aime,...}` — selects the JSON schema for parsing
  prompt / completion / pass-key from the lm-eval samples file.
- `--max-samples 200` — cap. HE has ~100 passing for a strong source;
  MBPP has 200-300; LCB-medium has 30-100.
- `--max-len 4096` — hard ceiling on per-sample tokens (still useful
  with chunking — caps runaway 32k AIME completions).
- `--skip-grad-patterns "embed_tokens,lm_head"` — these tensors are
  huge and downstream `--pr682-turbo` skips them anyway; saves VRAM
  and ~30% wall time.

## `competence_combine.py` — what to know

Combines **N tasks per source** into one map per source.

```bash
python competence/competence_combine.py \
    --map "src:humaneval:results.json:src__he.safetensors" \
    --map "src:mbpp:results.json:src__mbpp.safetensors" \
    --map "src:aime:results.json:src__aime.safetensors" \
    --raw-rate \
    --signal weight_taylor \
    --output-dir combined/
```

The per-task weight defaults to that task's pass-rate (`--raw-rate`),
e.g. HE@0.61 + MBPP@0.45 + AIME@0.27 → relative weights 0.46/0.34/0.20.
Rationale: a source that scores 0% on AIME contributes zero useful
gradient signal there; weighting by pass-rate prevents zero-rate tasks
from washing out informative ones.

The `--raw-rate` flag uses the literal pass rate. Without it, the
default is `softmax(rate)` which compresses the dynamic range (less
helpful when one source dominates a task).

After combine, the per-source maps are normalized so that
`sum_sources(map[k]) == 1` for every param element `k` — this is what
`omnimergekit.py --fisher` consumes.

## Production patterns

### Pattern A: bake AIME signal into an existing combined map

You already have a 3-source combined map from HE+MBPP and want to inject
AIME signal from the one source that solved AIME problems. Don't
re-extract HE+MBPP — blend mathematically:

```python
# pseudo-code from recipes/microcoder_4b/local_4b_competence_v2h.sh
W_OLD = HE_RATE + MBPP_RATE
W_NEW = W_OLD + AIME_RATE
new[k] = (W_OLD * old_combined[k] + AIME_RATE * aime_map[k]) / W_NEW
```

This is mathematically equivalent to a 3-task combine (because combined
is itself a normalized weighted sum), and saves the cost of
re-extracting HE+MBPP.

### Pattern B: stop a long sweep and resume later

`competence_extract.py` does NOT support resume yet (TODO). Workaround:
process tasks one at a time (`--task he`, then `--task mbpp`, then
`--task aime`) and combine afterward. If the AIME run dies mid-way, the
HE and MBPP outputs are intact.

### Pattern C: small VRAM (≤16 GB)

Reduce `--max-len` to 1024 and `--chunk-len` to 512. For Qwen3.5 4B this
fits in 12-14 GB. Signal quality is lower (less reasoning context per
sample), but on HE/MBPP where samples are short anyway, the difference
is invisible.

## Validated configurations

| Model | GPU | `--max-len` | `--chunk-len` | Notes |
|-------|-----|-------------|---------------|-------|
| Qwen3.5 4B | RTX 3090 24GB | 4096 | 1280 | Tested. Peak VRAM 8.5G between chunks. |
| Qwen3.5 4B | RTX 3090 24GB | 1280 | 0 (no chunking) | Tested. Peak ~22 GB; tight. |
| Qwen3.5 27B | A100 80GB | 4096 | 0 | Untested with chunking — single-shot fits. |
| Gemma 4 26B-A4B | A100 80GB | 4096 | 0 | Tested. MoE experts get per-expert grad allocation; ~50 GB peak. |
