# Experiments journal

Each entry summarizes what was attempted, what landed, and what was
rejected. Entries are not chronological — they're grouped by model
family. Numbers below come from real runs; the recipe scripts in
`recipes/` reproduce them.

## Qwen3.5-27B Omnimerge series

Sources: 4 fine-tunes of `Qwen/Qwen3.5-27B-Instruct` (Claude/Gemini
distills + creative + code teachers).

| ID | Method | GPQA-D | HE | MBPP | Notes |
|----|--------|--------|----|----|-------|
| **v2** (published) | `omnimerge_v2 --v2-features fisher,darex,emr,obim` | **69.19%** | wins 3/4 | wins 3/4 | Gold-standard recipe. `--weights` from Fisher. |
| v3a | Cross-base (Qwen3.6 base + 3 Qwen3.5 fine-tunes) | -- | -- | -- | Test of cross-base merging. Works; same recipe as v2. |
| v3b | Same-base on Qwen3.6 (3 Qwen3.5 sources retargeted) | -- | -- | -- | Reveals Qwen3.6 policy fragility — `<think>` token leak ~80% on samples. |
| v4 | Lower density 0.45, higher q 0.92 | regression | -- | -- | Over-pruning of small deltas. Rejected. |
| v4-mlp-skip | v4 + skip MLP gate-proj | regression | -- | -- | Disabled the very layers that hold the policy signal. Rejected. |

Recipe: `recipes/omnimerge_27b/pod_omnimerge_v2.sh`

## Qwen3.5-4B MicroCoder series

Goal: best 4B coder via 2-3 source merge. Sources: `jackrong-v2`
(reasoning), `continuum-code-forged` (code), `jackrong-python` (Python
specialist).

### Phase 1: 2-source DARE-TIES baselines (m4–m8)

Quick sweep of vanilla DARE-TIES + variants on 2 sources.

| ID | Method | HE | MBPP | LCB-30 | Verdict |
|----|--------|----|----|--------|---------|
| m4-v2 | DARE-TIES | -- | -- | -- | baseline |
| m6 | DARE-TIES + per-layer aware | -- | -- | -- | marginal |
| m6-hybrid | mix of m6 + retraining-style anneal | -- | -- | -- | parked |
| m7-detector | detector-based importance | -- | -- | -- | rejected |
| m7v2 / m7v3 | m7 layer-aware variants | -- | -- | -- | parked |
| m8 | aggressive density | -- | -- | -- | rejected |
| **m4-exlrp-v2** | `omnimerge_v2` with ex-LRP signal | -- | -- | -- | LRP signal too noisy on 4B — rejected |

### Phase 2: Fisher-based with competence maps (v2a–v2h)

| ID | Method | HE | MBPP | LCB-30 | GSM8K | MMLU-Pro | AIME | HE-Plus | Verdict |
|----|--------|----|----|--------|-------|----------|------|---------|---------|
| jackrong-v2 (source) | -- | -- | -- | -- | 83.0 | 56.8 | **26.7** | 54.9 | reasoning specialist |
| continuum-forged (source) | -- | -- | -- | -- | 79.0 | 49.1 | 0.0 | 48.2 | code specialist |
| jackrong-python (source) | -- | -- | -- | -- | 75.0 | 58.7 | 0.0 | 49.4 | python specialist |
| v1b | DARE-TIES baseline | -- | -- | -- | -- | -- | -- | -- | reference |
| v2a-d | Fisher variants | -- | -- | -- | -- | -- | -- | -- | iterating |
| **v2e-fisher-darex** | `omnimerge_v2 fisher,darex` (2 sources) | -- | -- | -- | 82.0 | 52.7 | 3.3 | 50.6 | best 2-source |
| **v2g-3src-fisher-darex** | 3-source extension of v2e | -- | -- | **26.67** | 81.0 | 53.0 | 0.0 | 49.4 | best LCB; AIME washed out |
| **v2h-3src-fisher-darex-aime** (in flight) | v2g + differential AIME signal blended in | -- | -- | -- | -- | -- | -- | -- | targeting AIME recovery + LCB retention |

Recipes:
- `recipes/microcoder_4b/local_4b_competence_finalize.sh` (v2g build + LCB)
- `recipes/microcoder_4b/local_4b_competence_v2h.sh` (v2h: AIME blend)
- `recipes/microcoder_4b/local_4b_extended_eval.sh` (extended eval)

Key learning from this series: **adding zero-AIME sources to a merge
drags AIME from 26.7% → 0%** even when the merge is otherwise dominated
by the AIME-capable source. The Fisher reweighting normalizes
*per-element* across sources, so on parameters where the AIME source's
gradient signal is large but the others' is near-zero, the merge still
gets significant contribution from the others, washing out the AIME
distinguishing direction. The v2h fix: build a separate AIME-only
differential map on the AIME source's wins, blend it into that source's
combined map at weight `AIME_pass_rate / (HE_pass_rate + MBPP_pass_rate
+ AIME_pass_rate)`.

## Gemma 4 26B-A4B compression series

Sources: `google/gemma-4-26B-A4B-it` (128 experts top-8).

| ID | Method | GPQA-D | Status |
|----|--------|--------|--------|
| 128e (original) | -- | **75.25%** | reference |
| 109e (drop 19/layer + router recal) | REAP baseline | **71.72%** | **published** |
| 98e (drop 30/layer + router recal) | REAP, more aggressive | ~67% | published |
| 109e + residual expert | REAP + packed dropped neurons | loops on chemistry | rejected (REAP Theorem 1) |
| 109e + DERN (k-means) | REAP + spherical k-means redistribution | broken impl | parked |
| 124e / 120e hybrid | drop fewer + manual hybrid expert | marginal gain | parked |
| 109e Wanda neuron-prune | structured per-neuron pruning | 5-10× worse | rejected |
| 109e SVD/MoE-I² | rank reduction | no size win | rejected |

Recipe: `recipes/gemma4/run_pruned_q6k_pipeline.sh`

## Cross-cutting learnings

These aren't experiments themselves but they shape every future run.

- **Differential vs collected Fisher**: collected Fisher (all docs) ≈
  uniform averaging in effect. Differential Fisher (only docs THIS source
  uniquely solved) actually preserves source distinguishing capability.
  This is the whole point of the competence pipeline.
- **Pass-rate as task weight in `competence_combine.py`**: weighting tasks
  by their pass rate (`--raw-rate`) prevents zero-rate tasks from
  contributing zero gradient signal that would dilute the combined map.
- **Linear blending into existing combined maps** (used in v2h) is
  mathematically equivalent to a 3-task combine, because combined is
  itself a normalized weighted sum. Saves re-extracting unchanged
  task signals.
- **HE/MBPP via `/v1/completions` on chat models breaks scoring**.
  Markdown fences cause `exec()` SyntaxError. Either use `/v1/chat/completions`
  or rescore with `eval/rescore_humaneval_strip_fences.py`.
- **Gemma 4 needs `--reasoning-budget 8192`** at llama-server start. Not
  optional. Without it the model emits malformed `<|channel>` tokens and
  lm_eval crashes mid-eval.
