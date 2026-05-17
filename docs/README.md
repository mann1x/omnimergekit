# omnimergekit / docs

This directory is the **research log** for omnimergekit's gemma4 + 4b
projects. Reading guide by track:

## Gemma 4 26B-A4B MoE pruning + recovery (current focus)

Chronological:

1. [`METHOD_gemma4_pruning.md`](METHOD_gemma4_pruning.md) — baseline method
   (v3 single-class, v4 max-across-5-classes). 109e + 98e-v3 + 98e-v4 era.
2. [`T17_v5_targeted_pruning_strategy.md`](T17_v5_targeted_pruning_strategy.md)
   — v5-targeted design: two-tier (synthetic + PASS-trace) per-class
   contribution scoring. Birth of v5-code / v5-science / v5-math variant
   families.
3. [`T19_v5fixed_sweep_results.md`](T19_v5fixed_sweep_results.md) — 13-variant
   aggregation sweep on v5-code (2026-05-16, 10h32m runtime). Full results
   table, cross-axis analysis, T18 candidate triangulation.
4. [`T19_he_chat_filter_fix.md`](T19_he_chat_filter_fix.md) — RCA + patch
   for the `extract_chat` filter bug that hid HE-1 wins on Gemma 4 reasoning
   output. Offline-rescore tool + in-module regression test.
5. [`T18_moe_router_recovery.md`](T18_moe_router_recovery.md) — Step 0/1/2/3
   framework for post-prune router rumination fix. Free-knob recovery,
   EAC-MoE TopK-MSE, DiffCal differential corpus.
6. [`T19_5_longer_smoke_results.md`](T19_5_longer_smoke_results.md) —
   longer-smoke (gsm8k_v128pass + humaneval_20 + lcb_medium_5) on the 4
   sweep candidates + v4 baseline anchor. Eliminates B3 and D1, lands B2
   at v4 math parity, motivates the v5-coder track (T21).

Supporting tables / data:

* [`T19_v5fixed_sweep_summary.tsv`](T19_v5fixed_sweep_summary.tsv) — final
  TSV from the sweep build script (note: `lcb_smoke` column is broken —
  use the omk_eval log line for ground-truth LCB-1; see HE filter fix RCA)
* [`T19_v5fixed_sweep_he1_rescore.json`](T19_v5fixed_sweep_he1_rescore.json)
  — per-variant HE-1 rescore output (true HE-1 column for the
  pre-fix variants)

## Gemma 4 31B dense head pruning

* [`HEAD_PRUNING_31B.md`](HEAD_PRUNING_31B.md) — head-prune recipe + heal
  results. Covers T7 / T8 arc.

## Competence-map (cross-model targeted merging)

* [`METHOD_competence_pipeline.md`](METHOD_competence_pipeline.md) — Tier-B
  pass-trace extraction + per-head / per-neuron competence maps + Fisher
  blending.

## Methodology / infrastructure

* [`METHOD_omnimerge_v2.md`](METHOD_omnimerge_v2.md) — omnimerge V2 recipe
* [`METHOD_kl_distillation.md`](METHOD_kl_distillation.md) — KL distillation
  recipe (used by microcoder_4b)
* [`CONDA_ENVS.md`](CONDA_ENVS.md) — env reqs / pins (note: omnimergekit
  conda env is the canonical default since 2026-05-16; see
  [memory/feedback_default_env_omnimergekit.md](../../memory/feedback_default_env_omnimergekit.md)
  on solidpc)
* [`EVAL.md`](EVAL.md) + [`../eval/EVAL_PROTOCOL.md`](../eval/EVAL_PROTOCOL.md)
  — eval suite design and omk_eval contract

## Recipes catalogue

Each recipe is a self-contained pipeline. See `recipes/<project>/`:

* `recipes/gemma4/` — v3 / v4 build + LCB-suite launchers (legacy 109e/98e era)
* `recipes/gemma4/v5_moe_sweep/` — **v5 sweep + T18 router recovery
  scripts** (current focus)
* `recipes/gemma4_31b/` — 31B dense head-prune machinery
* `recipes/microcoder_4b/` — distillation recipes (v2a..v2r ablation series)
* `recipes/omnimerge_27b/` — Qwen3.5-27B omnimerge V2

## Eval components

* `eval/omk_eval.py` — canonical eval entry point (per-bench templates;
  vLLM-first, llama.cpp fallback)
* `eval/templates/` — per-bench omk_eval YAML configs
* `eval/lm_eval_tasks/humaneval_chat/` — chat-aware shadow of stock
  humaneval (custom `extract_chat` filter — see HE filter fix RCA)
* `eval/lcb/` — LiveCodeBench shim + helper
* `eval/rescore_he1_smoke.py` — offline HE-1 rescore tool
