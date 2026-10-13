# gemma-4-A4B-20B-v9-agentic — recipe

> **CORRECTION 2026-10-13 — the recipe below is NOT the agentic-targeted v9-agentic.**
> `v8coder_fk12_drop_map.json` ("STD12") is built on the **v7 code map**
> (`expert_neuron_v7_code.json`): it is v7-coder with four experts swapped and uses no
> agentic traces. It was mislabelled as a v9-agentic variant. The agentic-targeted line is
> `maps/v9_agentic_drop_map.json` (data `expert_neuron_v9_agentic_code.json`, 43 multi-turn
> tool-calling sessions as tier B). Two candidates on that map, each adding the 12
> loop-specific pins, are being built by `scripts/build_v9_agentic_ab_solidpc.sh`; this
> page will be rewritten for whichever is selected. Nothing has been published to
> Hugging Face from the recipe below.

Published as [`ManniX-ITA/gemma-4-A4B-20B-v9-agentic`](https://huggingface.co/ManniX-ITA/gemma-4-A4B-20B-v9-agentic).
Internal build name: **STD12** (the K = 12 rung of the force-keep ladder). Successor of
`gemma-4-A4B-98e-v7-coder-it` (STD16, K = 16).

From this release the name carries the total parameter count (`20B`) instead of the
experts per layer (`98e`). The shape is unchanged: 98 of 128 experts in each of 30 layers,
20.5B parameters, 38.1 GB in bf16.

## The recipe in one screen

```
base        google/gemma-4-26B-A4B-it
drop map    maps/v8coder_fk12_drop_map.json          (30 experts dropped per layer)
              generator   scripts/generate_drop_map_v5fk.py
              data        expert_neuron_v7_code.json  (v7 competence map)
              target 98 · protect-top 16 · alpha 2.0 · strategy max · normalize rank
              breadth-bonus 0.5 · outlier-mode median · score-mode legacy
              class weights  math 1 · logic 1 · code 3 · science 1 · creative 1
                             multilingual 0 · humaneval 0 · humaneval+ 0 · lcb_medium_55 2
              force-keep  30 broad pins (maps/broad_fk.txt) + 12 loop-specific pins
apply       scripts/expert_drop.py
then        recipes/gemma4/v5_moe_sweep/router_shared_upweight.py --alpha 1.2 --target mlp.down_proj.weight
template    chat_template.jinja  (sha256 2e4c13d9…cbac, the agentic-loop fix)
no DERN fold
```

The 12 loop-specific pins (`layer:expert`, by falling specificity):

```
0:62 1:114 3:68 21:111 8:0 0:100 7:63 19:116 8:85 18:102 25:122 5:85
```

They are the top 12 of `traces/expert_neuron_v8_agentic_diff.json` by
specificity = loop mass ÷ clean mass. `scripts/gen_fk12_map.sh` re-derives that rule on
every run and refuses to continue unless top-8 and top-16 reproduce the published `fk8`
and `fk16` pin sets exactly, and unless a broad-pins-only run reproduces
`v8coder_fkbroad_drop_map.json`.

Against v7-coder (`scripts/v8coder_fk16_drop_map.json`) the map differs in four experts:

| layer | dropped in v9-agentic, kept in v7-coder | kept in v9-agentic, dropped in v7-coder |
|---:|---:|---:|
| 7 | 44 | 93 |
| 11 | 22 | 39 |
| 20 | 0 | 81 |
| 22 | 29 | 112 |

## Sampler

```
temperature 0.8 · top_k 64 · top_p 0.95 · repeat_penalty 1.08 · repeat_last_n 96 · presence_penalty 0.05
```

`generation_config.json` here is the file shipped in the HF repo (the subset
`transformers` reads). Greedy stays the cross-model anchor in the eval protocol; this
sampler is a separately named cohort.

## Scripts (all under `scripts/`)

| script | what it does |
|---|---|
| `gen_fk12_map.sh` | derives the pin rule, gates on reproducing fk8/fk16/fkbroad, writes the fk12 drop map |
| `build_std12_v5_model.sh` | bs2 build: drop + upweight, then Q6_K and Q4_K_M off one calib_v5 imatrix (the tiers used in hands-on testing) |
| `pod_v9_agentic_std12.sh` | pod build from the vendor base: drop + upweight, HF upload, AtomicChat imatrix |
| `fetch_std12_to_solidpc.sh` | delivery of the bs2 quants with sha check |
| `build_v9_agentic_map.sh`, `build_v9_dropmap.sh`, `build_v9_model.sh`, `build_v9_q4km.sh`, `build_v9_q40_gate.sh`, `v9_chain.sh`, `v9_classes.py`, `v9_final_report.py`, `run_lcb_v9.sh` | the September v9 candidate: an agentic competence map (2048-token tier A, 43 multi-turn tool-calling sessions as tier B) and its drop map. Superseded by the pin-ladder build above; kept because the map and its measurements are reused |
| `build_v9_ac_imatrix.sh`, `build_v9_ac_quants.sh` | AtomicChat imatrix arm of that candidate |
| `build_std27_v5_model.sh`, `build_std27_q4km.sh`, `run_lcb_std27.sh` | the 27-pin rung, rejected in agentic use |
| `traces_to_eog_corpus.py`, `trim_traces.py`, `manic_to_tierb.py`, `split_eog_corpus.py`, `rank_agentic_eog_lift.py`, `eog_map_overlap.py`, `eog_agreement_probe.py`, `compare_drop_maps.py`, `loop_table_split.py` | trace conversion and map comparison tools |
| `matrix_v9_agentic.json`, `matrix_v9_agentic_rp115.json` | sampler matrices for the frozen-conversation loop gate |

## maps/

| file | |
|---|---|
| `v8coder_fk12_drop_map.json` (+ `.summary.json`) | **the published drop map** (copy of `scripts/…`) |
| `v8coder_fkbroad_drop_map.json`, `broad_fk.txt` | the broad-pins-only base map and its 30 pins |
| `v9_agentic_drop_map.json` (+ `.summary.json`) | drop map of the superseded September candidate |

## traces/

| file | rows | what |
|---|---:|---|
| `expert_neuron_v8_agentic_diff.json` | 40 candidates | loop-vs-clean routing mass per dropped expert: the source of the pins |
| `expert_neuron_v8_agentic_diff.json.phase1_seqs.json` | 54 sequences | the token sequences behind it: 9 frozen agentic fixtures × seeds, 8 looped, 46 clean |
| `qset_FULL.json` | 43 sessions | resolved multi-turn SWE-bench-style sessions in 7 languages (38–285 messages) — tier B of the agentic competence map |
| `manic_v4_fixed_71_trim.json` | 71 sessions | agent sessions that fixed the manic_miner task, trimmed with `trim_traces.py` (tool output shortened, model output untouched) |
| `gemma4_control_11_trim.json` | 11 sessions | the same task solved by the unpruned Gemma 4 control |
| `eog_corpus_v4_t106.jsonl`, `eog_corpus_sweA_t49.jsonl` | 71 / 7 | the sessions above rendered through the chat template for an end-of-turn (`agentic_eog`) routing pass |

The traces contain public open-source repository content (file text, maintainer
addresses in headers) and no credentials; the repo's gitleaks gate runs over them.
