# T19.5 — Longer-smoke validation on T18 candidates + v4 baseline anchor

**Date**: 2026-05-16 (4-candidate run) + 2026-05-17 (v4 baseline + analysis)
**Wall**: 4 sweep candidates 8h59m (19:12 → 04:11) + v4 baseline 1h31m (04:12 → 05:43)
**GPU**: solidPC RTX 3090, single-GPU, vLLM NVFP4A16 backend
**Triad**: gsm8k_v128pass (95q) + humaneval_20 (20q) + lcb_medium_5 (5q),
all selected so 128e-NVFP4A16 PASSes every problem (so a variant FAIL
is signal — pruning damage — not noise — universally-hard problem).

## TL;DR

The T19 sweep's small-N smoke (gsm8k_30 + 1q HE + 1q LCB) was **misleading
on two of four candidates** and undersampled the variant-vs-variant trade
space. Longer-smoke landed three durable findings:

1. **B2_max_mathcode is at v4 baseline parity on math** (-4pp strict / +1pp
   flex). Publishable as a v5-math candidate as-is, no T18 needed.
2. **Every v5fixed sweep variant has large code damage vs v4** (HE-20 -35 to
   -80pp, LCB-5 -60 to -100pp). Class-weight tuning could not recover code
   because the structurally-pruned experts are gone — class weights modulate
   *which* axis bleeds most, not whether code bleeds.
3. **B3 ("code-pure pole") was a single-question fluke.** Its sweep-smoke
   HE-1 = LCB-1 = 1.0 was lucky; on the longer smoke it has silent-empty
   pathology at p10, HE-20 = 0.20, LCB-5 = 0.00. Eliminated.

The original assumption "recover math, code follows" is wrong. Math recovery
in B2 did not lift code. Next move: build a v5-coder using B2's strategy
(max + dual-class weighting) **with code as primary** instead of math. See
T21 follow-up below.

## Final results table

All five rows on the same vLLM recipe: `--reasoning-parser gemma4 +
default_chat_template_kwargs.enable_thinking=true +
thinking_token_budget=12288 + max_gen_toks=16384`, BF16 dtype, served via
omk_eval / `eval/templates/*.yaml`.

| variant | aggreg | weights | gsm strict | gsm flex | HE-20 | LCB-5 |
|---|---|---|---:|---:|---:|---:|
| **v4 baseline** (cd-max) | TF pooled | — | **0.853** | 0.916 | **1.000** | **1.000** |
| **B2_max_mathcode** | max | `3 1 2 1 1 1 1 1` | 0.811 | **0.926** | 0.45 | 0.40 |
| B4_max_broad | max | `2 1 3 2 1 2 2 2` | 0.589 | 0.632 | 0.65 | 0.20 |
| D1_lp4_mathcode | lp4 | `3 1 2 1 1 1 1 1` | 0.568 | 0.642 | 0.60 | 0.20 |
| B3_max_tgtheavy | max | `1 1 1 1 1 3 3 3` | 0.526 | 0.589 | 0.20 | 0.00 |

Class-weight slot order is `[math, logic, code, science, creative,
tgt_he, tgt_he+, tgt_lcb]` (per `scripts/expert_neuron_v5_code_fixed.json`).

### Anchored gaps (v4 = reference)

| variant | gsm strict | gsm flex | HE-20 | LCB-5 |
|---|---:|---:|---:|---:|
| **B2** | **−4 pp** | **+1 pp** | −55 pp | −60 pp |
| B4 | −26 | −28 | −35 | −80 |
| D1 | −28 | −27 | −40 | −80 |
| B3 | −33 | −33 | −80 | −100 |

## Selection criterion + question sets

Every problem in every bench is a **verified 128e-NVFP4A16 PASS** on the
current vLLM recipe. This makes a variant FAIL discriminative (pruning
damage), not noisy (universally-hard problem). The 128e samples used as
the gold filter are from
`eval_results_vllm_suite/128e/{gsm8k_100,humaneval_full,lcb_medium_55}/`,
runs of 2026-05-13.

* `gsm8k_v128pass` — 95 indices from the gsm8k_100 stride-13 sample,
  excluding the 5 indices where 128e failed on either strict-match or
  flexible-extract (excluded: 156, 195, 312, 546, 676).
* `humaneval_20` — 20 stride-from-128e-PASS indices spanning HE-164:
  `[0, 8, 16, 24, 33, 42, 50, 58, 66, 74, 83, 91, 99, 107, 115, 123, 131,
  140, 148, 156]`. 128e passes 163/164 of HE-164 (only HE/38 fails) — the
  stride doesn't hit HE/38.
* `lcb_medium_5` — first 5 medium problems since 2024-10-01, sorted by
  contest_date. By coincidence (verified) the first 5 — `lcb/leetcode/{3579,
  3566, 3502, 3593, 3629}` — are all 128e-PASS; first 128e-FAIL (3607) is at
  date-order position 6.

Templates: `eval/templates/{gsm8k_v128pass,humaneval_20,lcb_medium_5}.yaml`.

## Per-variant analysis

### B2_max_mathcode — v4 math parity, code damage

* **gsm: at v4 parity within noise.** −4pp strict, +1pp flex. The
  math-flex column actually beats v4 marginally (0.926 vs 0.916).
* **HE-20 0.45, LCB-5 0.40**: ~ −55/−60pp vs v4. Strong on math, half
  the code rate of v4.
* **Failure shape on gsm**: 35 questions are BOTH-fail (strict+flex);
  median fail-response 23k chars; p90 36k chars; 0 silent-empty; 1 short.
  Pure wrong-answer / rumination-loop failures. The model thinks for
  tens of thousands of tokens and lands on a wrong answer.
* **Verdict**: publishable as v5-math today. Class-weight slot order
  prioritized math (3×) and bought parity. Code class only at 2× was
  insufficient to preserve the code circuit.

### B4_max_broad — best HE, weak math, worst LCB

* gsm strict 0.589 / flex 0.632 — drops ~26-28pp vs v4. The "broad"
  weighting (`2 1 3 2 1 2 2 2`) couldn't preserve any single axis well.
* HE-20 0.65 — best among the four candidates but still −35pp vs v4.
* LCB-5 0.20 — only beats B3. The broad weight didn't help LCB.
* **Verdict**: dominated by B2 on math and LCB. Only wins HE-20, by
  +0.20 over D1. Weak T18 anchor.

### D1_lp4_mathcode — small-smoke #1, longer-smoke #3

* **Same weights as B2** (`3 1 2 1 1 1 1 1`), **different aggregation
  (`lp4` vs `max`)**.
* Same-weights B2 vs D1 shows aggregation alone moves gsm strict
  −24pp (0.811 → 0.568), gsm flex −28pp (0.926 → 0.642).
* HE-20 +15pp (0.45 → 0.60), LCB-5 −20pp (0.40 → 0.20).
* **The aggregation strategy is the load-bearing knob, not the
  weights.** `lp4` is `(Σ s^4)^(1/4)` — a soft max that's heavier on
  the largest score but doesn't pin to it. On v5fixed scores `lp4`
  evidently selects different boundary experts than `max`, with the
  boundary experts mattering far more than the per-class weight design.
* **T19's gsm8k_30 ranking misled here.** D1 was #1 on small-N gsm-
  strict (0.767) and #3 on longer-smoke (0.568). The boundary effect
  averages out across more questions.
* **Verdict**: strictly dominated by B4 on every column. Eliminated.

### B3_max_tgtheavy — fictional code specialist

* Sweep-smoke had B3 at HE-1 (rescored) = 1.0 + LCB-1 = 1.0, badged
  as the "code-pure pole" of the Pareto frontier.
* Longer-smoke: HE-20 0.20 (4/20), LCB-5 0.00 (0/5). End-to-end
  collapse.
* **Silent-empty pathology at p10**: omk_eval flagged
  `p10 completion length 5 < 60 chars` on B3 HE-20 — the 10th-percentile
  completion is 5 chars (`" eyes"`-style). At least 2/20 outright
  generation failures.
* **Mechanism**: weighting `1 1 1 1 1 3 3 3` (tier-B 3× exclusively)
  preserved very narrow code-trace experts at the cost of every other
  circuit, including the generic-code class itself. The narrow
  specialists empty out on prompts they weren't traced on.
* **Verdict**: eliminated. Sweep-smoke HE-1/LCB-1 were single-question
  flukes (B3 hit 1.0 on q=32 and q=3579 by lucky pattern match).

## Cross-axis findings

### 1. Aggregation strategy beats class weights at small drop budgets

Same weights `3 1 2 1 1 1 1 1`, different aggregation:
- max → B2: gsm 0.811 / 0.926, HE 0.45, LCB 0.40
- lp4 → D1: gsm 0.568 / 0.642, HE 0.60, LCB 0.20

Δ on gsm: −24pp strict. The aggregation function decides which
boundary experts get dropped; class weights modulate WHICH boundary
experts, but they can't compensate for choosing the wrong boundary.

### 2. Small-N smoke is misleading

T19 used `gsm8k_30 (first 30) + humaneval_1_smoke (HE/32) +
lcb_medium_1_smoke (lcb/leetcode/3579)`. Longer-smoke results
reordered the ranking:

| metric | T19 small-smoke #1 | longer-smoke #1 |
|---|---|---|
| gsm-strict | D1 (0.767) | **B2 (0.811)** |
| gsm-flex | B2 (0.833) | **B2 (0.926)** |
| HE (1q vs 20q) | B3+B4+B5+D3+D4 tied at 1.0 | B4 (0.65) |
| LCB (1q vs 5q) | B3+B4+B5+D3+D4 tied at 1.0 | B2 (0.40) |

D1 and B3 are misranked on small-N. Even B4 winning HE-20 doesn't mean
B4 should be the published code variant — 0.65 is far from a usable HE
score.

### 3. Math and code are NOT coupled in this pruning regime

The original assumption was: recover math, code follows (math reasoning
underpins code reasoning). The data contradicts:
- B2: math at v4 parity (0.811/0.926), HE only 0.45.
- B4: math far below v4 (0.589), HE only 0.65.
- B2 vs B4: math +22-29pp lifts HE by −20pp (B2 LOSES HE vs B4).

Mathematical-reasoning experts and code-emission experts are separable
under this pruning. A real v5-coder needs a code-targeted weight design,
not better math.

### 4. Even the best candidate is far from v4 on code

Best HE-20 in the sweep is B4 = 0.65 (−35pp vs v4 1.00). Best LCB-5 is
B2 = 0.40 (−60pp vs v4 1.00). The 30-experts-per-layer drop budget
combined with the class-weight knob alone cannot reach v4's code
performance from a v5fixed score distribution. Two paths forward:
either reweight the score generation to favor code from the start, or
apply T18 router recovery to recover the dropped contribution.

## Decisions for T18 and beyond

### T18 router recovery — narrowed candidate set

The original T18 plan targeted four candidates {B2, B4, D1, B3}.
Longer-smoke results narrow this to:

- **B2** — minimal T18 (Step 1 sanity check), publishable on math as-is.
- **B4** — most code headroom; if T18 is to demonstrate router
  recovery value, B4 is the candidate where the delta will be largest.
- **D1, B3** — eliminated.

### T21 — v5-coder C1_max_codetb (active follow-up)

Apply B2's dual-class weighting strategy WITH CODE AS PRIMARY:

```
class-weights:  "1 1 3 1 1 2 2 2"
                 ^ ^ ^ ^ ^ ^ ^ ^
                 | | | | | | | +-- targeted_lcb_medium_55  (Tier-B 2× secondary)
                 | | | | | | +---- targeted_humanevalplus  (Tier-B 2× secondary)
                 | | | | | +------ targeted_humaneval       (Tier-B 2× secondary)
                 | | | | +-------- generic_creative           (1×)
                 | | | +---------- generic_science            (1×)
                 | | +------------ generic_code               (3× — PRIMARY)
                 | +-------------- generic_logic              (1×)
                 +---------------- generic_math               (1×)
```

This is the mirror of B2's `3 1 2 1 1 1 1 1` — same dual-class
weighting pattern, just rotated to make code primary instead of
math. Hypothesis: preserves the generic-code circuit (3×) PLUS the
Tier-B PASS-trace observations (2×) as second-order support, instead
of B3's exclusive Tier-B over-weighting which produced narrow
specialists that empty out.

**Success floor** (HE/LCB recovery test):
- HE-20 ≥ 0.80 (B4's 0.65 + meaningful lift, halving the gap to v4)
- LCB-5 ≥ 0.60 (B2's 0.40 + meaningful lift, halving the gap to v4)
- gsm-strict ≥ 0.65 (must not collapse below B4's 0.589)

Build script: `recipes/gemma4/v5_moe_sweep/build_v5coder_c1.sh`.
Eval: `recipes/gemma4/v5_moe_sweep/longer_smoke_v5coder.sh`.
Results table will be appended to this doc when C1 lands.

## Files

* `eval/templates/{gsm8k_v128pass,humaneval_20,lcb_medium_5}.yaml` —
  selection-criterion templates.
* `recipes/gemma4/v5_moe_sweep/longer_smoke_t18_candidates.sh` — runs
  the 4-candidate sweep.
* `recipes/gemma4/v5_moe_sweep/longer_smoke_v4_baseline.sh` — runs the
  v4 baseline anchor.
* `recipes/gemma4/v5_moe_sweep/build_v5coder_c1.sh` — builds v5-coder
  C1 candidate.
* `recipes/gemma4/v5_moe_sweep/longer_smoke_v5coder.sh` — runs the
  v5-coder longer smoke.
* `eval_results_vllm_suite/v5fixed_t18_longer_smoke/` — per-variant
  per-template raw results.

## Cross-references

* [T19_v5fixed_sweep_results.md](T19_v5fixed_sweep_results.md) — the
  13-variant aggregation sweep that produced the 4 candidates.
* [T19_he_chat_filter_fix.md](T19_he_chat_filter_fix.md) — the
  `extract_chat` filter fix that surfaced the silent-empty pathology
  at p10 on B3.
* [T18_moe_router_recovery.md](T18_moe_router_recovery.md) — Step
  0/1/2/3 framework, now scoped to {B2, B4} candidate pair.
