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

### Sources (each Q6_K, served via llama-server)

| Source | HE | MBPP | LCB-30 | GSM8K | MMLU-Pro | AIME-30 | HE+ |
|--------|----|----|--------|-------|----------|---------|-----|
| `jackrong-v2` (reasoning) | 60.4 | 45.0 | 23.3 | **83.0** | 56.8 | **26.7** | **54.9** |
| `continuum-code-forged` (code) | 59.2 | **53.4** | 13.3 | 79.0 | 49.1 | 0.0 | 48.2 |
| `jackrong-python` (Python) | 57.3 | 47.0 | 20.0 | 75.0 | **58.7** | 0.0 | 49.4 |

Each source wins on different axes: jackrong-v2 reasoning, continuum-forged
MBPP/code-throughput, jackrong-python MMLU-Pro/general. Goal: a merge that
keeps each source's strengths.

### Phase 1: 2-source DARE-TIES variant sweep (v1b, m2–m8)

Quick exploration of vanilla DARE-TIES, layer-aware, and detector-based
importance on 2 sources. Numbers omitted where the variant was rejected
on a quick sniff test (sample inspection or HE pass rate alone).

| ID | Method | HE | MBPP | LCB-30 | Verdict |
|----|--------|----|----|--------|---------|
| **microcoder-v1b** | DARE-TIES baseline | 61.0 | 46.2 | 10.0 | reference baseline |
| m2-turbo | DARE-TIES + density 0.6 | -- | -- | -- | parked |
| m4-v2-dare | layer-aware DARE | -- | -- | -- | marginal |
| m6 / m6-hybrid | per-layer + anneal | -- | -- | -- | parked |
| m7-detector / m7v2 / m7v3 | detector-based importance | -- | -- | -- | rejected (signal noisy) |
| m8 | aggressive density 0.4 | -- | -- | -- | rejected |
| m4-exlrp-v2 | `omnimerge_v2` with ex-LRP signal | -- | -- | -- | LRP signal too noisy on 4B |

### Phase 2: Fisher-based with competence maps (v2a–v2h)

All run with `omnimergekit.py --method omnimerge_v2 --v2-features fisher,darex`,
`density=0.53 --darex-q=0.85`. Variants differ in source set, fisher
restriction, and density.

| ID | Sources | HE | MBPP | LCB-30 | GSM8K | MMLU-Pro | AIME-30 | HE+ | Verdict |
|----|---------|----|----|--------|-------|----------|---------|-----|---------|
| v2a-competence | 2 (no diff) | 61.0 | 46.2 | -- | -- | -- | -- | -- | early calibration |
| v2b-competence-diff | 2 (diff maps) | 61.0 | 46.2 | -- | -- | -- | -- | -- | confirms baseline |
| v2c-fisheronly-diff | 2 (fisher only) | 56.7 | 51.6 | -- | -- | -- | -- | -- | code↑, reasoning↓ |
| v2d-fisher-d030 | 2 (density 0.30) | 57.9 | 51.4 | -- | -- | -- | -- | -- | density too aggressive |
| **v2e-fisher-darex** | 2 (jv + cf) | **59.2** | 52.6 | 23.3 | 82.0 | 52.7 | 3.3 | 50.6 | **best 2-source** |
| **v2g-3src-fisher-darex** | 3 (+ jp) | 56.1 | **54.0** | **26.67** | 81.0 | 53.0 | 0.0 | 49.4 | **best LCB**, AIME washed out |
| **v2h-3src-fisher-darex-aime** | 3 + AIME diff signal blended | 56.1 | 51.4 | **26.67** | **81.0** | 53.3 | **0.0** | 49.4 | **rejected** — strictly worse than v2g (-2.6 MBPP, AIME unmoved) |

**Headline observation:** going from v2e (2 sources) to v2g (3 sources)
won LCB (+3.4 pp) but lost AIME (3.3 → 0.0). v2h tried to recover AIME
via a focused differential map on the 8 AIME problems jackrong-v2
uniquely solved. **It did not work.** AIME stayed at 0/30, LCB held at
26.67, MBPP slipped 2.6 pp. Every other axis flat to within ±0.3 pp.

**Why the v2h fisher-blend failed:**

8 winning AIME problems gave too sparse a gradient signal to redirect
the merge toward AIME-solving capability. Per-element fisher importance
identifies which params jackrong-v2 used to solve those 8 problems, but
biasing the merge weights on those params (via blend `(1.05·old +
0.27·aime_jv) / 1.32`) only marginally tilts those params toward jv —
not enough to overcome the averaging-toward-base effect on the
AIME-distinguishing direction in parameter space. The MBPP loss
indicates the AIME signal did pull params away from their code-utility
configuration, but didn't compensate with usable AIME ability.

**What would actually work for AIME recovery (future):**

1. **SFT distillation** of jackrong-v2's AIME outputs (or its full
   reasoning corpus) into the merge — cleanly transfers the
   problem-solving direction with proper gradient density.
2. **Direct interpolation toward jackrong-v2** on the layers that hold
   the reasoning policy (likely `mlp.gate_proj` mid-layers, given
   Qwen3.6 token-leak experiments showed policy lives there). Skip
   fisher entirely on those layers; pass jv straight through.
3. **More extraction data** — collect AIME-style problems from open
   datasets (MATH, NuminaMath), eval jackrong-v2 on them, extract
   fisher on the larger winning set. With 30-50 docs the signal might
   become dense enough to bias the merge meaningfully.

**Verdict:** **v2g remains the published candidate** for the 3-source
MicroCoder line. v2h is logged as a negative result.

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
