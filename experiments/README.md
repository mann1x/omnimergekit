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
| **v2i-jv-base-task-arith** | jv as merge_base, task-vectors of cf+jp from Qwen3.5-4B | **57.3** | 52.0 | **26.67** | **83.0** | 52.5 | **3.3** | **50.0** | **competitive** — matches jv-source GSM8K exactly, AIME non-zero, HE+1.2 / HE+ +0.6 / MBPP −2.0 vs v2g |

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

**Verdict (v2h):** v2g remained the better merge against v2h. v2h is
logged as a negative result.

### v2i: task-arithmetic merge (jackrong-v2 as base, deltas vs Qwen3.5-4B)

After v2h's failure, we changed strategy entirely. Instead of trying to
extract a stronger AIME signal via fisher, we used the task-arithmetic
formulation (Ilharco et al., ICLR 2023):

```
merged = jackrong-v2 + 0.55 · DARE(cf − Qwen3.5-4B) + 0.45 · DARE(jp − Qwen3.5-4B)
```

`jackrong-v2` is the merge_base — passes through unchanged at full
strength (no DARE drop, no fisher dilution). The cf and jp task vectors
are computed FROM the shared common ancestor (Qwen3.5-4B) so they encode
"add code skill" / "add python skill" cleanly, without simultaneously
"undoing reasoning". This required an `omnimergekit.py` patch to add
`--task-base` (commit `f6fde8d`).

**Result vs v2g:**

| Δ | HE | MBPP | LCB | GSM8K | MMLU-Pro | AIME | HE+ |
|---|---|---|---|---|---|---|---|
| v2i − v2g | +1.2 | **−2.0** | = | **+2.0** | −0.6 | **+3.3** | +0.6 |

Notably, **v2i matches the jackrong-v2 source GSM8K exactly (83.0)** —
a behavior v2g/v2h didn't achieve (both 81.0). Conciseness is also
preserved (median chars on GSM8K: jv 229, v2i 239, v2g/v2h 259-262).
AIME recovered 1/30 (3.33%) — small but non-zero, vs v2g/v2h's 0.0.
LCB ceiling held at 26.67. Cost: MBPP −2.0 vs v2g.

**Verdict (v2i):** Better balanced profile. The task-arithmetic
formulation does what it promises — preserves the merge_base's behavior
in directions orthogonal to the source task vectors. For a "code +
reasoning balance" model card, **v2i is the better candidate.** For a
"code-only" framing where MBPP is the headline, v2g still leads by 2 pp.

**Methodology lesson:** when one source dominates a capability axis the
others are zero on, the natural framing is task-arithmetic, not
symmetric multi-source merging. The published v2g recipe assumed all
three sources contributed *some* signal in every subspace; v2i drops
that assumption for the axes where it's false.

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

Sources: `google/gemma-4-26B-A4B-it` (128 experts top-8, 30 layers).
Drop targets: 128→109 (~15% drop) and 128→98 (~23% drop).
Quants: GGUF Q6_K for distribution, NVFP4A16 (ModelOpt) for vLLM eval.

### Early structural attempts (legacy)

| ID | Method | GPQA-D | Status |
|----|--------|--------|--------|
| 128e (original) | -- | **75.25%** | reference |
| 109e v3 (drop 19/layer + router recal) | REAP baseline | **71.72%** | **published** |
| 98e v3 (drop 30/layer + router recal) | REAP, more aggressive | ~67% | published |
| 109e + residual expert | REAP + packed dropped neurons | loops on chemistry | rejected (REAP Theorem 1) |
| 109e + DERN (k-means) | REAP + spherical k-means redistribution | broken impl | parked |
| 124e / 120e hybrid | drop fewer + manual hybrid expert | marginal gain | parked |
| 109e Wanda neuron-prune | structured per-neuron pruning | 5-10× worse | rejected |
| 109e SVD/MoE-I² | rank reduction | no size win | rejected |

### v4 — class-min protected drop (98e, published)

Method: per-layer score = wnorm × α + tc per task class (math/logic/code/
science/creative), drop bottom-30 with per-class minimum-protect to prevent
specialist starvation. NVFP4A16 quant.

| Bench (full) | 128e | **v4 NVFP4A16** | Δ |
|---|---|---|---|
| gpqa_diamond | 75.25% | 69.19% | −6.1pp |
| gsm8k_100 | 0.91 | **0.93** | +2pp |
| math500_100 | 0.94 | 0.93 | −1pp |
| aime_30 | 0.43 | 0.40 | −3pp |
| humaneval_full | 0.95 | 0.92 | −3pp |
| humanevalplus_full | 0.92 | 0.91 | −1pp |
| ifeval_100 | 0.87 | 0.86 | −1pp |
| arc_challenge | 0.92 | 0.91 | −1pp |
| lcb_medium_55 | 0.89 (49/55) | 0.78 (43/55) | −11pp |

LCB is the weak axis: v4 ruminates to length-cap on ~10% of LCB problems
(mean 13,178 completion tokens, p90 hits 16,384). gsm8k tokens are tight
(mean 76, p50 61). Code (HE+) is in v4-anchor territory. Verdict: v4 is
publishable but leaves significant LCB headroom on the table.

Recipe: `recipes/gemma4/run_pruned_q6k_pipeline.sh`, model card:
`https://huggingface.co/ManniX-ITA/gemma-4-A4B-98e-v4-it`.

### v5-coder layer-relevance series (in flight, 2026-05-15..17)

Goal: recover the v4 LCB gap by varying the per-layer floor (i.e.
per-layer `protect_top` budget). Each variant differs only in the
floor-map JSON. All NVFP4A16, same v5_code Tier-B signal, n=98 budget.
Smoke triad: gsm8k_30 + humanevalplus_curated30 + lcb_medium_55.

| ID | Floor-map design | gsm8k_30 strict | HE+/30 | LCB-55 | mean tok (gsm) | Verdict |
|----|------------------|----------------:|-------:|--------:|---------------:|---------|
| C1 | `max_codetb` (pure Tier-B max) | catastrophic | -- | -- | **6,111** | rejected — code-only starves math |
| C2 | `v4floor80_breadth50` | partial | -- | -- | 2,893 | partial — still ruminates |
| C5 | `v4floor95_breadth50` (73 swap slots, guardrail) | -- | -- | -- | -- | parked |
| **C6** | **`v4floor_perlayer_breadth50`** (per-layer top98_mean) | 25/30 | 28/30 | **48/55 (87.27%)** | **71** | **publishable winner** |
| C7 | `v4floor_perlayer_classmin_logic18_creat22` | 24/30 strict, 26/30 flex | -- | -- | ~3,500 | rejected — class-min forces specialist demotion |
| C8 | `v4floor_perlayer_userprofile` (smooth gradient, L0=95 L25=70 trough) | -- | -- | -- | -- | rejected |
| C9-steeper | linear L0=95→L29=65 (540 slots) | -- | -- | -- | ~3,800 | rejected — confirmed C6 win is NOT monotonic descent |
| C9-flatter | linear L0=90→L29=80 (390 slots) | -- | -- | -- | ~3,800 | rejected — same as steeper |
| C10-gini | per-layer floors scaled by gini coefficient | 25/30 (matches C6) | -- | -- | **3,919** | rejected — matches C6 accuracy but ruminates 55× |

**Headline finding**: C6 is the only variant that preserves both
accuracy AND compact reasoning. C6 LCB-55 = 87.27% (+9.1pp vs v4) at
12,796 mean tokens (~v4's 13,178). HE+/30 28/30 ≈ v4 anchor. gsm 25/30
is −1 problem vs v4 anchor (closed by T18-shared, see below).

The shape of the C6 floor (per-layer top98_mean of the v4 ranker) is
NOT linear — it has an L10 spike to 92, an L25 trough to 70, and a
characteristic non-monotonic curve. **All C9-style linear approximations
of the C6 shape broke rumination**, even when they preserved the L0=95
and L29 endpoints. The floor design must follow the *data signal*
(top98_mean of v4's per-class ranker), not any human-priors gradient.

Recipes: `recipes/gemma4/v5_moe_sweep/build_v5coder_c{1,2,5,6,8}.sh`,
`build_v5coder_c9_{steeper,flatter}.sh`, `build_v5coder_c10_gini.sh`.

### T18 — Step 1 router free-knobs (post-prune recovery)

Three post-prune router edits that don't need re-quantization or retraining:
1. `router_topk_dial.py` — change top-k from 8 to N (tested N=6)
2. `router_shared_upweight.py` — scale shared expert by α_shared (tested 1.2)
3. `router_soft_transfer.py` — k-NN soft redistribution of dropped router
   rows (tested α=0.3, top-k=3)

**T23 — combined stack on v4** (clean model, all 3 knobs at defaults):

| Bench | v4 anchor | v4 + T18 combined | Δ |
|---|---|---|---|
| gsm8k_30 strict | 26/30 (0.867) | **18/30 (0.600)** | **−8 problems** |
| gsm8k mean tok | 76 | **5,738** | **75× rumination** |

Verdict: combined stack DESTROYS a clean model. T18 is not a free win.
The knobs were designed for already-ruminating routers, not healthy ones.

**T30 — single-knob isolation on C6** (2026-05-17, in flight):

| Knob | gsm8k_30 strict | gsm8k flex | mean tok (gsm) | Decision |
|------|----------------:|-----------:|---------------:|----------|
| baseline C6 | 25/30 (0.833) | 25/30 | 71 | — |
| topk-6 only | 23/30 (0.767) | 27/30 (0.900) | **3,864** | **STOP (rumination, same as v4 T23)** |
| shared α=1.2 only | **26/30 (0.867)** | **26/30** | 76 | gsm/HE+ flat ± 1 problem, **LCB-55 in flight** |
| transfer α=0.3 only | -- | -- | -- | queued (solidpc, 17:50 launch) |

Shared α=1.2 isn't reducing tokens overall on gsm/HE+ — at C6's already-
tight token budget there's nothing to fix. The interesting signal is on
LCB-55 (where v4 ruminated to 13k mean tok): **first 6 problems show
shared α=1.2 converts a length-cap failure (16,384 tok) into a clean
stop (12,544 tok)** while tightening variance across problems. If the
pattern holds, the knob is a routing stabilizer that kills length-cap
failures specifically. Verdict pending full LCB-55.

Top-k 6 alone on C6 reproduces the v4 T23 rumination signature exactly
(3,864 mean tok = 50× C6 baseline, strict drops). Top-k dial is rejected
on Gemma 4 regardless of underlying model.

Recipes/scripts:
- `recipes/gemma4/v5_moe_sweep/router_{topk_dial,shared_upweight,soft_transfer}.py`
- `recipes/gemma4/v5_moe_sweep/apply_t18_step1.sh` (combined, legacy)
- `scripts/apply_t18_single_knob_c6.sh` (isolation runner, T30)
- Docs: `docs/T18_moe_router_recovery.md`

### 62e aggressive prune + capability redistribution (PARKED 2026-06-03)

Pushed the prune to **62e** (A2 = `gemma-4-A4B-62e-fc15_25-p8-pes120-it`, ~12 GB
VRAM) and asked whether the experts dropped at 62e can be **redistributed** back
into the survivors. Full log — selection-exhaustion (T175/T187-189), closed-form
fold (HC-SMoE/MergeMoE/REAM) and trainable KD (E-RankProbe/E-ExpertKD/code-KD),
on both diffuse-multilingual and localized-code drivers — in
**[`gemma4_62e_redist.md`](gemma4_62e_redist.md)**.

Verdict: **no redistribution lever beats the A2 baseline.** A fixed router can't
independently gate folded function (REAP Thm 1); trainable KD reconstructs (ML) or
noise-trades (code, +0.6 pp HE+ but constrained-loops double). Capability recovery
at a fixed budget is a **selection** problem (v5/v6-coder beat A2 by keep-set
choice), not a redistribution problem. Council-confirmed structural + publishable
(`csl-2026-06-02-2050-c4f7`). A2 stays unshipped pending a budget/scope decision.

### Open questions / next steps

1. LCB-55 verdict on C6+shared (in flight, ETA 18:10 CEST 2026-05-17).
   Is the length-cap-conversion pattern real or a 1-in-6 fluke?
2. C6+transfer isolation (just launched solidpc). The transfer knob
   makes the most theoretical sense (dropped-router-row k-NN soft
   redistribution) but burned hard in T23 combined.
3. Full canonical 9-bench on C6 (or C6+shared, if it wins LCB) — replicate
   v4's model card structure for publish-ready comparison.
4. If C6+shared wins LCB: publishable recipe `gemma-4-A4B-98e-v5coder-C6-shared`.

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
