# T19 — v5fixed aggregation sweep on Gemma 4 26B-A4B (98e)

**Date**: 2026-05-16 (run: 08:13:47 → 18:45:37 CEST, **10h 32m** wall-clock,
solidpc RTX 3090).
**Base**: `google/gemma-4-26B-A4B-it` (128 experts, top-8 routed + 1 shared
mlp per MoE layer).
**Target**: 98 experts (drop 30 per layer, `protect_top=16`), quantized to
NVFP4A16 via modelopt 0.43.0.
**Sweep dimension**: 13 variants exploring the **aggregation strategy ×
class-weight** product space for the multi-class
contribution-to-drop-rank mapping built on top of
`expert_neuron_v5_code_*.json` (the T17-spec Tier-A+Tier-B-PASS data).
**Smoke per variant**: `gsm8k_30` + `humaneval_1_smoke` (HE/0) +
`lcb_medium_1_smoke` (1 medium LeetCode problem).

## Why this sweep

After v4 (max-across-classes, p16, top‑8) the published 98e‑v4 was strong on
math but lost code; the smoke‑first orchestrator (tb10 / tb15) confirmed the
choice of `max + uniform weights` was tunable but unstable. Mean was tested
earlier and lost; the user pushed back on collapsing back to mean and asked
for a full sweep across **strategies AND weights AND protect knobs** before
moving on to **T18 router recovery** (post-prune rumination fix). The goal is
not a single winner — it is a **map of the trade-off surface** so that T18 is
applied to multiple distinct profiles, not just one.

## Sweep design

The drop-map score per (layer, expert) is built from `expert_neuron_v5_code_*.json`:

```
score[layer, expert] = aggregate_strategy(
    weights[class] × per_class_score[class, layer, expert]
    for class in 8 classes
)
```

Each class is one of the 5 Tier-A categories (math / logic / code / science /
creative) plus 3 Tier-B PASS-trace categories. Per-class scores are
unit-mean-normalised per layer to be commensurate across classes.

### Axis 1 — aggregation strategy (Group A, uniform weights)

Each strategy picks the per-expert score from the [class × expert]
vector at a layer:

| Tag | Strategy | Formula |
|-----|----------|---------|
| A2_lp4_uni | `lp4` | `(Σ_c w_c · s_c^4)^(1/4)` — L4 norm |
| A3_top3mean_uni | `top3mean` | mean of the top-3 weighted classes |
| A4_softmax_t4_uni | `softmax_t4` | smooth-max @ τ=4 (log-sum-exp) |
| A5_second_uni | `second` | second-highest weighted class (drops the max) |

(`max` strategy at uniform weights is implicit in v4; not re-run.)

### Axis 2 — class weighting (Group B, `max` strategy)

Re-use of `max` strategy as the anchor; vary per-class weight vector:

| Tag | Profile | Weight vector (math, logic, code, science, creative, t6, t7, t8) |
|-----|---------|-------------------------------------------------------------------|
| B1_max_genheavy   | first-5 emphasis  | `2 2 2 2 2 1 1 1` |
| B2_max_mathcode   | math+code emphasis | `3 1 2 1 1 1 1 1` |
| B3_max_tgtheavy   | last-3 emphasis | `1 1 1 1 1 3 3 3` |
| B4_max_broad      | broad up-weight, code peak | `2 1 3 2 1 2 2 2` |
| B5_max_xtgt       | extreme last-3 | `1 1 1 1 1 5 5 5` |

### Axis 3 — strategy × weights crossover (Group D)

Pre-flight overlap matrix showed the protect-axis Group C (p20, floor=2,
psum) produced 30/30 overlap with the baseline on this data — degenerate.
Replaced with Group D crossovers of "interesting strategy × interesting
weight":

| Tag | Strategy × weights |
|-----|--------------------|
| D1_lp4_mathcode      | `lp4`      × mathcode `3 1 2 1 1 1 1 1` |
| D2_second_mathcode   | `second`   × mathcode |
| D3_top3mean_genheavy | `top3mean` × genheavy `2 2 2 2 2 1 1 1` |
| D4_second_tgtheavy   | `second`   × tgtheavy `1 1 1 1 1 3 3 3` |

`protect_top` held at 16 for every variant; `floor=0`, `pstrat=same`.

## Full results table

Smokes: gsm8k_30 (30 q), HE-1 (HE/0 = `has_close_elements`), LCB-1 (1
medium LeetCode problem ≥ 2024-10-01).

Both the **as-reported** TSV values and the **HE-rescored** + **true LCB**
values are shown. The TSV's `lcb_smoke` column is broken — the outer build
script's LCB score parser doesn't read `pass_at_1` correctly and writes 0
for every row. The omk_eval log line `[omk_eval HH:MM:SS] score: X.X` is the
ground truth (and is the value below in the `LCB` column).

The `HE` column shows the rescored value using the fixed `extract_chat`
filter (`utils_chat.py` rewrite, 2026-05-16). The original
`humaneval_chat/extract_chat` was broken on Gemma 4 reasoning output — every
variant scored 0 because the filter chopped at the first stray ```; see
[T19_he_chat_filter_fix.md](T19_he_chat_filter_fix.md).

| Tag                  | Strategy   | Weights         | gsm_str | gsm_flex | **HE-1** | **LCB-1** | Profile           |
|----------------------|------------|-----------------|--------:|---------:|---------:|----------:|-------------------|
| A2_lp4_uni           | lp4        | uniform         |   0.633 |    0.733 |      0.0 |       n/a | uniform          |
| A3_top3mean_uni      | top3mean   | uniform         |   0.600 |    0.700 |      0.0 |       0.0 | uniform          |
| **A4_softmax_t4_uni**| softmax_t4 | uniform         |   0.633 |    0.733 |  **1.0** |       0.0 | HE-only          |
| A5_second_uni        | second     | uniform         |   0.600 |    0.667 |      0.0 |       0.0 | uniform (weakest)|
| B1_max_genheavy      | max        | gen-heavy       |   0.600 |    0.667 |      0.0 |       0.0 | weak             |
| **B2_max_mathcode**  | max        | mathcode        |   0.633 |  **0.833** |    0.0 |       0.0 | **math-flex leader** |
| B3_max_tgtheavy      | max        | tgtheavy        |   0.533 |    0.567 |  **1.0** |   **1.0** | code-only        |
| **B4_max_broad**     | max        | broad           | **0.667** |  0.733 |  **1.0** |   **1.0** | **all-rounder** ★ |
| B5_max_xtgt          | max        | xtgt (5×)       |   0.567 |    0.600 |  **1.0** |   **1.0** | code-only (extreme) |
| **D1_lp4_mathcode**  | lp4        | mathcode        | **0.767** |  0.767 |    0.0 |       0.0 | **math-strict leader** ★ |
| D2_second_mathcode   | second     | mathcode        |   0.600 |    0.700 |      0.0 |   **1.0** | LCB-only         |
| D3_top3mean_genheavy | top3mean   | gen-heavy       |   0.500 |    0.667 |  **1.0** |       0.0 | HE-only (weakest math) |
| D4_second_tgtheavy   | second     | tgtheavy        |   0.567 |    0.633 |  **1.0** |       0.0 | HE-only          |

`★` = candidate for T18 router recovery.

A2 LCB ran before the LCB-1 omk_eval template fix landed; its true LCB-1
needs a re-run alongside the longer-smoke pass.

## Cross-axis analysis

### Weight axis dominates math preservation

Math (gsm8k_30) varies by class-weight more than by strategy:

```
mathcode 3·1·2·1·1·1·1·1   → 0.633–0.833 flex (B2, D1, D2; D1 → 0.767 strict, best)
uniform                    → 0.667–0.733 flex
gen-heavy 2·2·2·2·2·1·1·1  → 0.500–0.667 flex
tgt-heavy 1·1·1·1·1·3·3·3  → 0.567–0.633 flex
broad 2·1·3·2·1·2·2·2      → 0.733 flex
xtgt 1·1·1·1·1·5·5·5       → 0.600 flex
```

Class 1 (math) at 3× and class 3 (code) at 2× is the strongest math signal.
Down-weighting them (tgt/xtgt) hurts. Gen-heavy (first-5 at 2×) hurts.
The "broad" profile (B4) keeps classes 1+3 elevated AND elevates classes 6/7/8 — best balance.

### Last-3 classes preserve code

Every variant that passes BOTH HE-1 and LCB-1 has weight ≥ 2× on the
last-3 classes (`{B3, B4, B5}` all have last-3 ≥ 3 except B4 which has 2):

| Tag | last-3 weights | HE | LCB |
|-----|---------------:|---:|---:|
| B3 (tgt 3×)   | 3·3·3 | 1.0 | 1.0 |
| B4 (broad)    | 2·2·2 | 1.0 | 1.0 |
| B5 (xtgt 5×)  | 5·5·5 | 1.0 | 1.0 |
| D4 (second × tgt) | 3·3·3 | 1.0 | **0.0** |

So upweighting classes 6–8 is necessary but not sufficient — the
**strategy** still has to be strong enough (max ≥ second). The Tier-B
PASS-trace categories (classes 6–8 in this v5_code mapping) are the
**code-preservation** signal.

### Math-emphasis weights are anti-code at `max`/`lp4` strategies

Every `mathcode` × {max, lp4} variant fails HE-1 (B2, D1). With weaker
`second` strategy + mathcode, LCB-1 surprisingly passes (D2 = 1.0) while
HE-1 still fails. The strategy modulates which code bench survives.

### Single-q smokes are noisy but the pattern is consistent

13 variants × 1 HE-1 + 1 LCB-1 is low signal — pass@1 = 1/1 has wide CI. But
the consistency of the math axis (strict_match correlates tightly across
all 13) and the consistency of "tgt weights + max strategy → code passes"
(B3, B4, B5 all 1+1) strongly suggests the pattern is real, not random.
Longer smokes (gsm8k_100, HE-20, LCB-5) on the chosen T18 candidates will
confirm.

## Three-tier candidate profile for T18

The sweep reveals a **trade-off triangle** instead of a single winner:

```
                       math-strict (D1_lp4_mathcode)
                              0.767 / 0.767 / HE 0 / LCB 0
                                       /\
                                      /  \
                                     /    \
                                    /      \
                          (B4_max_broad)    (B2_max_mathcode)
                          all-rounder ★     math-flex leader
                          0.667 / 0.733 / HE 1 / LCB 1   0.633 / 0.833 / HE 0 / LCB 0
                                    \    /
                                     \  /
                                      \/
                             code-pure (B3 / B5 max tgtheavy)
                             0.533 / 0.567 / HE 1 / LCB 1
```

**Top-3 picks for T18 router recovery** (see [T18_moe_router_recovery.md](T18_moe_router_recovery.md)):

1. **B4_max_broad** — strongest all-rounder; balanced math + both code benches passing
2. **B2_max_mathcode** — best math preservation; failing code is the rumination signal T18 should fix
3. **B3_max_tgtheavy** — pure-code reference; included even with weakest math because the user explicitly asked to compare router recovery's effect on a code-leaning variant

D1 is included as an additional math-strict comparison if T18 has budget.

## Artifacts

| Artifact | Path |
|---|---|
| Sweep launcher | [`recipes/gemma4/v5_moe_sweep/build_98e_v5fixed_sweep.sh`](../recipes/gemma4/v5_moe_sweep/build_98e_v5fixed_sweep.sh) |
| Smoke-first orchestrator | [`recipes/gemma4/v5_moe_sweep/build_98e_v5fixed_smoke_first.sh`](../recipes/gemma4/v5_moe_sweep/build_98e_v5fixed_smoke_first.sh) |
| Drop-map generator (v5) | [`gemma4/expert_pruning/generate_drop_map_v5.py`](../gemma4/expert_pruning/generate_drop_map_v5.py) |
| Neuron analysis (v5-fixed) | [`gemma4/neuron_analysis/expert_neuron_analysis_v5_targeted.py`](../gemma4/neuron_analysis/expert_neuron_analysis_v5_targeted.py) |
| Sweep summary TSV (final) | [`T19_v5fixed_sweep_summary.tsv`](T19_v5fixed_sweep_summary.tsv) |
| HE-1 rescore JSON | [`T19_v5fixed_sweep_he1_rescore.json`](T19_v5fixed_sweep_he1_rescore.json) |
| HE filter fix (RCA + patch) | [`T19_he_chat_filter_fix.md`](T19_he_chat_filter_fix.md) |
| Post-sweep picker | [`recipes/gemma4/v5_moe_sweep/pick_t18_variants.sh`](../recipes/gemma4/v5_moe_sweep/pick_t18_variants.sh) |
| T18 router recovery design | [`T18_moe_router_recovery.md`](T18_moe_router_recovery.md) |
| T17 v5-targeted strategy (origin) | [`T17_v5_targeted_pruning_strategy.md`](T17_v5_targeted_pruning_strategy.md) |
| Diff-corpus build | [`recipes/gemma4/v5_moe_sweep/build_diff_corpus.py`](../recipes/gemma4/v5_moe_sweep/build_diff_corpus.py) |

Raw eval results stay on the solidpc backup_models disk at
`eval_results_vllm_suite/v5fixed_sweep/<variant>/...` (not in this repo —
samples_*.jsonl files are 100s of MB total).

## Reproduction

```bash
cd /srv/.../backup_models  # any project root with model dirs

# 1. Build the 8-class neuron-analysis JSON for v5-code
python gemma4/neuron_analysis/expert_neuron_analysis_v5_targeted.py \
    --tier-b-corpus scripts/extract_pass_traces.py.output \
    --out scripts/expert_neuron_v5_code.json

# 2. Run the sweep
bash recipes/gemma4/v5_moe_sweep/build_98e_v5fixed_sweep.sh

# 3. After completion, pick top-N
bash recipes/gemma4/v5_moe_sweep/pick_t18_variants.sh 3
```

Per-variant wall-clock is ~40–50 min on a single RTX 3090 (drop-map +
expert_drop + NVFP4A16 quant + 3 smokes). Full 13-variant sweep is one
overnight run.

## Status — 2026-05-16 18:45 CEST

* **Sweep: DONE.** All 13 variants in TSV. 12 of 13 with valid LCB-1
  result (A2's pre-fix LCB needs re-run).
* **HE filter: FIXED.** `extract_chat` rewrite in `utils_chat.py` (in-tree,
  6-case self-test, AST-parse-trim + smart-dedent + always-prepend-prompt);
  all 7 completed variants offline-rescored without re-inference.
* **T18 chain: PLUMBED.** `router_topk_dial.py`, `router_shared_upweight.py`,
  `router_soft_transfer.py` (Step 1) ready; `router_eac_calibrate.py`
  (Step 2 EAC-MoE TopK-MSE) ready; `build_diff_corpus.py` produced a
  423-example / 103,915-token differential calibration corpus.
* **Diff corpus: STAGED.** `logs/diff_corpus.txt` on solidpc (HumanEval
  115 + HumanEval+ 104 + gsm8k_100 134 + math500_100 4 + ifeval_100 63 +
  ifeval_full 3, length-filtered to ≤ 70th percentile per bench).

## Next planned items

1. **Longer-smoke validation** on candidates {B4, B2, B3, D1}:
   gsm8k_100, HumanEval-20 (chat task), LCB-medium-5.
   Sanity-check the single-q results; promote to T18 only those that hold.
   - Re-run A2_lp4_uni LCB-1 to fill the missing data point.
2. **T18 Step 1** (free knobs) per candidate: `apply_t18_step1.sh <tag>`
   with sweep over `{topk=4,5,6} × {α_shared=1.0,1.2,1.5} × {α_transfer=0.0,0.3,0.5}`.
   Smoke each variant on gsm8k_30 + HE-1 + LCB-1; rank by composite.
3. **T18 Step 2** (EAC-MoE TopK-MSE) A/B per candidate, both corpora:
   `--corpus-file` flag toggled between {WikiText-2 baseline, diff_corpus.txt}.
   Compare which calibration source produces better post-step results.
4. **Step 3 decision gate** based on Step 2 deltas: if (B4 or B2) recovers ≥ +5pp
   on HE/LCB while keeping math, ship as 98e-v5; otherwise iterate.

## Cross-references

* [METHOD_gemma4_pruning.md](METHOD_gemma4_pruning.md) — pruning method overview
  (v3 / v4 baseline)
* [T17_v5_targeted_pruning_strategy.md](T17_v5_targeted_pruning_strategy.md) —
  v5-code targeted-pruning design
* [T18_moe_router_recovery.md](T18_moe_router_recovery.md) — Step 0/1/2/3 framework
* [T19_he_chat_filter_fix.md](T19_he_chat_filter_fix.md) — HE chat extractor RCA
* [EVAL_PROTOCOL.md](../eval/EVAL_PROTOCOL.md) — eval suite + omk_eval contract
