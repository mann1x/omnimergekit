# Gemma-4 26B-A4B — SWE-bench agentic competence maps (q-set convergence study)

Six competence maps built from **SWE-bench multi-turn agentic traces**, plus the
measurement that says how many traces per language a targeted map actually needs.

Built with `scripts/expert_neuron_analysis_v5_targeted.py` via
`recipes/gemma4/run_qset_maps.sh` on an RTX PRO 6000, bf16, 2026-09-15.

## What the files are

| file | traces | languages | note |
|---|---|---|---|
| `competence_gemma4_a4b_swebench_qsetA.json` | 7 | 7 × 1q | independent sample |
| `…qsetB.json` | 7 | 7 × 1q | disjoint from A |
| `…qsetC.json` | 7 | 7 × 1q | disjoint from A, B |
| `…qsetD.json` | 7 | 7 × 1q | disjoint from A, B, C |
| `…qsetE.json` | 12 | 4 × 3q | javascript, ruby, java, php |
| `…qsetF.json` | 12 | 4 × 3q | disjoint from E |

Languages are the SWE-bench log-parser families: `swe_cpp_c`, `swe_go`, `swe_java`,
`swe_javascript`, `swe_php`, `swe_ruby`, `swe_rust`. Traces are agent runs that
**resolved** their instance (gold-verified), carrying the model's own
`reasoning_content` — 69–90% of the token volume — which reaches the forward pass
only under the generation-time chat template with `preserve_thinking=True`.

Any two maps can be combined with `scripts/merge_competence_maps.py`: the stored
`wnsq/rnsq/wsum/tc/cc/neuron_act` are raw sums, so merging disjoint q-sets is
arithmetically identical to profiling them in one pass. `wnorm`/`rnorm` are the
intensive per-token RMS `sqrt(wnsq/tc)` and are recomputed, never summed. This is
the same mechanism that combines the LCB and MPE maps in the Qwen program.

## How many traces per language does a targeted map need?

Set-difference is the wrong endpoint. At a fixed drop-count the cut lands mid-
distribution, so experts ranked 30th and 31st swap under any perturbation — churn
with a floor that never reaches zero. The endpoint that matters is what the
disagreement **costs**: using the other sample's drop map, how much more competence
do you discard than you meant to? (`scripts/qset_drop_regret.py`.)

Calibrated against a **cross-language control** — map A compared against itself with
the language categories permuted (rust scored by go's data, etc.). That is the scale
of a genuinely different language, and the ceiling any sampling error can be worth.
Identity control reads exactly 0.000%.

Single basis, the 4 languages with enough traces for a 3q point:

| sample | REL. regret | vs cross-language |
|---|---|---|
| 1q vs 1q (A–B) | 151.6% | 1.10× |
| 1q vs 1q (C–D) | 298.1% | 2.16× |
| 2q vs 2q (AC–BD) | 82.9% | 0.60× |
| **3q vs 3q (E–F)** | **65.7%** | **0.48×** |
| *cross-language control* | *138.2%* | 1.00× |

**At 1q the map is not measuring the language.** Two independent samples of the same
language disagree *more* than two different languages do — the map is measuring which
repository the trace came from. 2q crosses below the control; 3q reaches roughly half
of it, and the gain per added trace has collapsed (−63% for 1q→2q, −21% for 2q→3q).

**Practical rule: 3q per language is the knee. 1q is actively misleading.** Going
further has sharply diminishing returns, and the current 43-trace PASS pool caps the
scarcest language (rust) at 4 anyway.

Rank agreement moves the same way — Spearman ρ 0.947/0.957 at 1q → 0.962 at 2q →
0.977 at 3q — but ρ alone cannot be read without the control, because the
cross-language control also sits at ρ 0.92–0.94.

## Caveats

- **Tier-A was skipped.** The `generic_*` categories are imported unchanged from the
  May-15 v5-code map (`--load-tier-a-from`), identical in every file, so they are
  carried once on merge rather than summed. These maps are built for the targeted
  comparison; a production drop map should run Tier-A itself or reuse a current one.
- **The imported `generic_*` cells carry NaN in `wsum`** — 152 cells per category,
  layers 11–29, inherited from that May-15 source. Inert: no drop-map scorer reads the
  map's `wsum` field (every `wsum` in the generators is a local weight-sum variable).
  All targeted categories produced by the current pipeline are clean.
- **`neuron_act` is stripped** (`[]`), matching the published Qwen maps. Verified
  field-by-field: every other value is unchanged across all 46,080 cells, so any
  scorer that does not read `neuron_act` produces an identical drop map. Neuron-level
  work (DERN-style redistribution) needs the full ~480 MB maps, kept on disk.
- Per-trace `set` labels read `C` — that is the pipeline's **weight class** (uniform
  weight 1.0 here), not the q-set letter. The two letter namespaces collide.
