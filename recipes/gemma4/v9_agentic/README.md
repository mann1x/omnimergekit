# gemma-4-20B-v9-agentic — recipe

Published as [`ManniX-ITA/gemma-4-20B-v9-agentic`](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic)
(bf16) and [`ManniX-ITA/gemma-4-20B-v9-agentic-GGUF`](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic-GGUF)
(39 tiers + F16, imatrix), and on ollama as `mannix/gemma-4-20B-v9-agentic`. Benchmarks and the
tool-calling results are on the model card; this page is the build.

Gemma 4 26B-A4B pruned from 128 to 98 experts in each of its 30 layers (20.45B parameters,
~4B active), selected on an agentic competence map. The name carries the total parameter count
(`20B`); earlier releases in this line were named by experts per layer (`98e`).

## Reproduce the drop map

```bash
PY=/path/to/omnimergekit-env/python bash recipes/gemma4/v9_agentic/reproduce_drop_map.sh
```

CPU only, about a minute. It rebuilds map A, ranks the multilingual rescues and rebuilds the
published map from the data in this directory, and stops unless each step matches what was
shipped — the last one byte for byte (`maps/gemma-4-20B-v9-agentic_drop_map.json`,
sha256 `84f2bd961772adbd…`).

## The recipe in one screen

```
base        google/gemma-4-26B-A4B-it
data        competence/expert_neuron_v9_agentic_code.slim.json   (agentic competence map)
generator   recipes/gemma4/generate_drop_map_v5fk.py
              target 98 · protect-top 16 · alpha 2.0 · strategy max · normalize rank
              breadth-bonus 0.5 · score-mode legacy · outlier-mode median (thresh 10000)
              class weights   generic math 1 · logic 1 · code 3 · science 1 · creative 1
                              targeted_swe java · rust · javascript · ruby · go · php · cpp_c  2 each
              v4 floor        scripts/v4_layer_floor_map_v7.json on competence/expert_neuron_base_v7.slim.json
              baseline        scripts/teacher_force_98e_p16_clean.json
              force-keep      120 end-of-turn lift pins + 12 loop pins          → map A
                              + 8 multilingual rescue pins                        → this build
apply       scripts/expert_drop.py
then        recipes/gemma4/v5_moe_sweep/router_shared_upweight.py --alpha 1.2 --target mlp.down_proj.weight
template    chat_template.jinja — Google's official Gemma 4 template, unmodified (sha256 ae53464bf3be2580…)
no DERN fold
```

## Steps

**1. Competence map.** `scripts/expert_neuron_analysis_v5_targeted.py` on the unpruned 128e,
2,048-token windows, answer channel (thinking off), traces rendered with
`templates/trace_chat_template.jinja`. Tier A: 2,048-token single-turn prompts for five generic
classes plus one software-engineering class per language (Java, Rust, JavaScript, Ruby, Go, PHP,
C/C++). Tier B: 43 resolved multi-turn tool-calling sessions with reasoning
(`traces/qset_FULL.json`). Classes and weights are emitted by `scripts/v9_classes.py`.
Driver: `scripts/build_v9_agentic_map.sh`.

**2. Selection** with the generator arguments above; the full call is in
`reproduce_drop_map.sh` and `scripts/v9ml_post.sh`.

**3. End-of-turn lift pins (120).** The top 4 experts per layer by routing lift on the end-of-turn
token `<turn|>` (id 106), over 71 agent sessions rendered through the chat template
(`traces/eog_corpus_v4_t106.jsonl`, ranked by `scripts/rank_agentic_eog_lift.py`):
`maps/candidates/eog_v4_t106_lift_pins.txt`.

**4. Loop pins (12).** Routing mass on looped vs clean runs of 9 frozen agentic fixtures
(`traces/expert_neuron_v8_agentic_diff.json`); specificity = loop mass ÷ clean mass, top 12:

```
0:62 1:114 3:68 21:111 8:0 0:100 7:63 19:116 8:85 18:102 25:122 5:85
```

Steps 1–4 give **map A** (`maps/candidates/A_lift120_plus_fk12.json`). It was picked over
candidate B (30 broad pins + the same 12, `B_broad30_plus_fk12.json`) on Q4_K_M test builds
(`scripts/build_v9_agentic_ab_solidpc.sh`).

**5. Multilingual / IFEval retargeting.** Map A has no multilingual or instruction-following
class. Two Tier-A classes were added on the same model, prompt length and template
(`scripts/v9ml_map.sh`, `--tier-a-extra`):

| class | prompts | source |
|---|---:|---|
| `generic_multilingual` | 95 | every IFEval "respond only in ⟨language⟩" prompt in rows 100–540, plus native-language prompts weighted to the Indic and Persian scripts |
| `generic_ifeval` | 48 | IFEval rows 100–540 with hard format constraints |

Both files are in `traces/calib_v9ml/` (built by `scripts/build_v9ml_calib.py`). IFEval rows
0–99 — the scored benchmark — are never used for calibration.

`scripts/merge_tier_a_categories.py` merges the two classes into map A's data only after the
re-measured generic classes reproduce map A's (per-layer Spearman ≥ 0.95 and top-16 overlap ≥ 13
in every layer; measured: an exact reproduction). Result:
`competence/expert_neuron_v9ml_code.slim.json`.

The swap was chosen from a ladder of minimal swaps (K = 2 to 16) after quick tests on map A's failing IFEval prompts. `scripts/v9ml_ladder.py`
ranks the experts map A drops by their per-layer rank in `generic_multilingual` (ties broken by
`generic_ifeval`), and adds the top K as force-keep pins. The generator then evicts the
lowest-scoring survivor in each of those layers by map A's own class scores. K = 8:

```
rescued  0:9   8:122  15:64  15:106  23:116  26:16  27:48  29:25
evicted  0:23  8:96   15:3   15:38   23:48   26:108 27:106 29:26
```

8 of the 2,940 kept experts change; everything else is map A.

**6. Build and publish** (pod, llama.cpp b9700):

1. `scripts/pod_v9ml_k8_imatrix.sh` rebuilds the bf16 (drop → α=1.2 → official template), gated on
   the evaluated shards. It then computes the imatrix on the AtomicChat corpus rendered through the
   official template: all 9,590 chunks, `--parse-special`, imatrix sha256 `820aab4a06031984…`.
2. `scripts/pod_v9ml_publish.sh` uploads the bf16 and builds every tier with
   `scripts/quantize_gguf.py --no-kld`. Each tier gets a load-and-answer check after upload, and the
   publish ends only once every file on the hub matches this run's sha256.
3. `scripts/pod_v9_xollama_publish.py` creates the ollama tags, only for tiers whose check passed. Every tag is asked three questions
   before the push and read back from the registry after it.

## Sampler

```
temperature 1.0 · top_k 64 · top_p 0.95 · min_p 0 · repeat_penalty 1.06 · repeat_last_n 96 · presence_penalty 0.025
```

`generation_config.json` is the file shipped in the HF repo (the subset `transformers` reads).
Benchmarks decode greedy with the same three penalties
(`eval/models/gemma-4-v9-agentic.yaml`, profile `greedy_pen`). With no profile flag, omk_eval
refuses to run this model (`require_profile`).

## Files

### maps/

| file | |
|---|---|
| **`gemma-4-20B-v9-agentic_drop_map.json`** (+ `.summary.json`) | **the published drop map** (= `v9ml_ladder/ml_k8.json`) |
| `candidates/A_lift120_plus_fk12.json` | map A — steps 1–4 |
| `candidates/B_broad30_plus_fk12.json` | candidate B, not selected |
| `candidates/eog_v4_t106_lift_pins.txt` | the 120 end-of-turn lift pins |
| `v9ml_sweep/` | first approach to step 5: the new classes as weighted classes (multilingual/ifeval weights 0/0, 1/1, 2/2, 2/0, 0/2), plus `A_repro` — map A regenerated from the merged data |
| `v9ml_ladder/` | step 5: `pins.txt` and every rung — `shared_k2…k16` (ranked on both classes), `ml_k8` (multilingual-first, shipped), `if_k8`, `ml_k3`, `ml_k8v` |
| `v9_agentic_drop_map.json`, `v8coder_fk12_drop_map.json`, `v8coder_fkbroad_drop_map.json`, `broad_fk.txt` | earlier maps of this line, not shipped |

### competence/

The competence maps without the per-neuron activation arrays (`scripts/slim_competence_map.py
--verify`: every other field of every cell is unchanged, and the generator never reads that
array). The full maps (0.3–0.7 GB) are kept off-repo.

| file | |
|---|---|
| `expert_neuron_v9_agentic_code.slim.json` | map A's data: Tier A + Tier B |
| `expert_neuron_v9ml_code.slim.json` | the same + `generic_multilingual`, `generic_ifeval` |
| `expert_neuron_base_v7.slim.json` | the v7 base map the per-layer v4 floor is ranked on |

### traces/ and templates/

| file | |
|---|---|
| `calib_v9ml/multilingual_calib_v9ml.jsonl`, `calib_v9ml/ifeval_calib_v9ml.jsonl` | the two step-5 classes |
| `qset_FULL.json` | Tier B: 43 multi-turn SWE-bench-style sessions in 7 languages |
| `eog_corpus_v4_t106.jsonl`, `eog_corpus_sweA_t49.jsonl` | agent sessions rendered for the end-of-turn routing pass |
| `manic_v4_fixed_71_trim.json`, `gemma4_control_11_trim.json` | the sessions behind them (tool output shortened, model output untouched) |
| `expert_neuron_v8_agentic_diff.json` (+ `.phase1_seqs.json`) | loop-vs-clean routing mass: the source of the 12 loop pins |
| `templates/trace_chat_template.jinja` | the template the competence traces were rendered with (sha256 `d9f21aac…`); the model ships `chat_template.jinja` |

The traces contain public open-source repository content and no credentials; the repo's
gitleaks gate runs over them.

### results/v9ml/

`v9ml_map.log` and `v9ml_post.log` (the step-5 map and merge/sweep/ladder runs) and
`router_diff_v9fail.json`, a diagnostic on map A's IFEval misses (routing mass of the experts
map A drops, on failing vs passing answers).

### Scripts (`scripts/`)

| script | step |
|---|---|
| `expert_neuron_analysis_v5_targeted.py`, `build_v9_agentic_map.sh`, `v9_classes.py` | 1 — competence map, classes and weights |
| `rank_agentic_eog_lift.py`, `traces_to_eog_corpus.py`, `trim_traces.py`, `split_eog_corpus.py` | 3 — end-of-turn lift pins |
| `build_v9_agentic_ab_solidpc.sh` | map A vs B test builds |
| `build_v9ml_calib.py`, `v9ml_fetch_then_map.sh`, `v9ml_mk_aria_in.py`, `v9ml_map.sh` | 5 — class prompts and their map |
| `merge_tier_a_categories.py`, `v9ml_post.sh`, `v9ml_same_map.py`, `v9ml_report.py`, `v9ml_ladder.py`, `v9ml_make_k3_k8v.py` | 5 — merge gate, sweep, ladder |
| `v9ml_quick_build.sh`, `v9ml_quick_eval.sh`, `v9ml_quick_eval_p1.sh`, `v9ml_quick_table.py`, `v9ml_shacheck.py`, `pod_v9ml_build.sh`, `pod_v9ml_eval.sh` | 5 — quick no-imatrix Q6_K builds and IFEval probes of the rungs (`eval/templates/ifeval_v9fail10.yaml`) |
| `router_diff_v9fail.py`, `v9ml_rdiff_smoke.py` | 5 — router diagnostic |
| `pod_v9ml_k8_imatrix.sh` (`_2gpu`), `pod_v9ml_k8_q.sh`, `pod_v9ml_k8_iq4nl.sh`, `pod_v9ml_k8_suite.sh` | 6 — bf16 rebuild, imatrix, eval quants, benchmark suite |
| `pod_v9ml_publish.sh`, `quantize_gguf.py`, `gguf_kld.py`, `pod_v9pub_reupload.py`, `pod_v9pub_resanity.py` | 6 — HF publish and tier checks |
| `pod_v9_xollama_publish.py` | 6 — ollama tags |
| `pod_v9pub_toolbench.sh` | tool-calling benchmark (`eval/toolbench/`) |
