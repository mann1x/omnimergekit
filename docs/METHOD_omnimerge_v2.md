# Method: `omnimerge_v2`

The `omnimerge_v2` method in `omnimergekit.py` composes four ingredients,
toggleable via `--v2-features`:

| Feature | Toggle | What it does |
|---------|--------|--------------|
| `obim` | OBIM-lite | Outlier-Bounded Importance Magnitude. Shrinks the dynamic range of the importance signal so a single outlier neuron can't dominate the mask. |
| `darex` | DAREx-q | Quantile-based DARE. Drops a `1-density` fraction of delta tensor elements, but uses the q-th quantile (`--darex-q 0.85` default) of `|delta|` rather than the mean to decide what's "small". Robust to long-tailed source distributions. |
| `emr` | EMR election | Election-style Median Rescue. When two sources agree on a sign and a third disagrees, the third is rescued at reduced weight only if its magnitude exceeds the disagreeing pair's median. Reduces averaging-towards-base on contested directions. |
| `fisher` | Fisher reweighting | Per-element importance from per-source competence maps replaces the user-supplied `--weights`. See [`METHOD_competence_pipeline.md`](METHOD_competence_pipeline.md) for how the maps are produced. |

`omnimerge_v2 --v2-features fisher,darex` is the recipe used in the
published Qwen3.5-27B-Omnimerge-v2 and the 4B MicroCoder series.

## Pipeline (per parameter tensor)

For each parameter `name` shared by base + N sources:

1. Compute deltas `Δ_i = source_i[name] - base[name]` for each source.
2. **DAREx-q** (if enabled): compute `|Δ_i|`, take `q`-quantile threshold,
   zero out elements below threshold, rescale survivors by `1/density`
   to preserve expected magnitude.
3. **Fisher reweight** (if enabled): per-element source weights come from
   the loaded competence maps, normalized across sources. If absent, fall
   back to `--weights` flag.
4. **EMR election** (if enabled): pre-mix sign-agreement vote; majority-sign
   keeps full weight, contested sources downweighted unless magnitude
   exceeds median.
5. **OBIM-lite shrink** (if enabled): clip the per-element importance vector
   `w_i` to a percentile range to prevent any one source's outlier from
   washing out the others on this tensor.
6. Combine: `merged[name] = base[name] + Σ_i w_i * Δ_i`.
7. **PR-682 turbo** (if `--pr682-turbo`): skip `embed_tokens` and `lm_head`
   from the per-element math entirely (they're huge and the merge math
   degrades them) — pass the base tensor through unchanged.

## Where each piece comes from

- DARE / TIES baseline: [Yu et al. 2024, "Language Models are Super Mario"](https://arxiv.org/abs/2311.03099); [Yadav et al. 2023, "TIES-Merging"](https://arxiv.org/abs/2306.01708).
- DAREx (quantile variant) and OBIM-lite (outlier-bounded importance):
  developed in this kit. The naming "OBIM-lite" reflects that it's a
  simplified one-pass version of the full OBIM (importance Mahalanobis
  shrink) — the lite version uses percentile clipping which empirically
  matched OBIM on the 27B sweep at a fraction of the cost.
- EMR (Election Median Rescue): adapted from mergekit's TIES sign-election
  into a continuous (rather than hard 1/0) downweight for the losing side.
- Fisher reweighting: the per-element source weights are the per-source
  fisher signal restricted to docs that source uniquely solved
  (differential maps). See competence pipeline doc.

## Hyperparameter cheat sheet

These were validated on Qwen3.5-27B (4 sources) and Qwen3.5-4B (2-3 sources):

| Flag | Default | Sane range | Notes |
|------|---------|-----------|-------|
| `--weights` | even split | `0.2-0.5` per source, sum normalized | Replaced by Fisher when `fisher` feature enabled. |
| `--density` | 0.5 | 0.4-0.7 | DARE drop fraction. Higher density = closer to plain weighted average. |
| `--darex-q` | 0.85 | 0.7-0.95 | Quantile of `|Δ|` used as drop threshold. Higher = drops more aggressively. |
| `--seed` | 42 | any | Affects DARE element-drop pattern (Bernoulli mask). |
| `--skip-patterns` | empty | comma-list of substrings | Tensors whose names contain any pattern pass through from base. Common: `"model.visual,mtp.layers"` for multimodal/MoE-prediction. |
| `--pr682-turbo` | false | flag | Skip `embed_tokens` + `lm_head` from per-element math. **Recommended for any merge >7B**, where `lm_head` is several GB. |

## When NOT to use `omnimerge_v2`

- **Cross-base merges** where one source is a different base architecture
  than the other sources (e.g. Qwen3.6 base + 3 Qwen3.5 fine-tunes).
  This works but the `<think>` policy can be fragile (see
  `experiments/qwen3_6_merge_policy_fragility.md`).
- **Single-task specialization.** If you want a code-only merge from a code
  source + math source, plain DARE-TIES with weights tilted toward the
  code source often beats Fisher because the math source's competence map
  contributes signal you don't want.
- **Compression-first goals.** This method preserves base shape; it does
  not reduce parameter count. For that, see `gemma4/expert_pruning/`.

## Walked-down failure modes

- **Markdown fences in HE/MBPP outputs after merge.** The merged model can
  inherit reasoning-mode fence-wrapping from one source while the scorer
  was set up for the other. Symptom: `pass@1=0` while inspecting samples
  shows technically-correct code wrapped in ` ```python ... ``` `. Fix:
  rescore with `eval/rescore_humaneval_strip_fences.py`.
- **Token leak (`<think>` artifacts) on Qwen3.6 same-base merges.** The
  policy lives in `mlp.gate_proj` layers 27-52, NOT in `lm_head` /
  `embed_tokens` (those are byte-identical to base after merge). Skip-patterns
  on those gate layers, or skip Fisher and use static `--weights` only,
  reduces leak to <5%.
- **`intermediate_size % 32 != 0`** on MoE experts → Q4_K / Q8_0
  silently fall back to F16 for those tensors and the GGUF balloons.
  Always check before quantizing: `python -c "import json; c=json.load(open('config.json')); print(c.get('moe_intermediate_size', c.get('intermediate_size'))); assert _ % 32 == 0"`.
