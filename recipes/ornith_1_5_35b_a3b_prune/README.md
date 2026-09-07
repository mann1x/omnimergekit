# Ornith-1.5-35B-A3B — 256e → 184e expert prune (Coder / CoderX)

Prunes `ManniX-ITA/Ornith-1.5-35B-A3B` (256 experts, 40 MoE layers + MTP head) to 184
experts per layer. Two arms ship from the same competence map:

| arm | drop map | selection |
|---|---|---|
| **Coder** | `results/drop_map_184e_coder_tc.json` | competence `tc` / `wmax`, targeted LCB+MPE channels at 1.5× |
| **CoderX** | `results/drop_map_184e_hybrid_p24.json` | same, plus `make_hybrid_dropmap --protect 24` (force-keep top-24 by REAP saliency) |

`--protect 24` does not add capacity — it **reallocates** it. Both arms drop 72/256 per
layer, but 24 of those 72 differ on every one of the 40 layers: **960 experts that Coder
keeps, CoderX evicts, and vice versa.**

Merge engine is `ream/omk_ream_merge.py` with `--saliency reap --merging none` (armJ
recipe, byte-identical to the Qwen3.6-35B-A3B recipe's copy). `--merging none` discards
evicted experts; nothing is folded.

## Pipeline

| step | script | notes |
|---|---|---|
| P3b | `p3b_competence_targeted.sh` | competence map over the merged router-calibration corpus (807 docs / 1.12M tokens) |
| P5.1 | `runhost/dump_reap_saliency.py` | REAP saliency dump |
| **P5.1b** | **`runhost/build_eog_emit_map.py`** | **terminator emit-map — see below** |
| P5.2 | `runhost/recover_keepsets.py` | keepsets (armD recovered from built weights, armE derived from the dump) |
| P5.3 | `runhost/make_hybrid_dropmap.py` | hybrid map at `--protect 24` |
| **P5.3b** | **`runhost/eog_keepset_gate.py`** | **EOG keepset gate — see below** |
| P5.4 | `ream/omk_ream_merge.py` | build, `--merging none` |
| P5.5 | `ream/verify_arm_identity.py` | hard identity gate |

Driver: `p5_armB.sh`.

## The EOG terminator gate (added 2026-09-07)

**Why.** Terminator health is invisible in the score. An arm that loses the experts
carrying routing mass at end-of-generation positions overruns, emits a malformed
thinking channel, the server's chat parser rejects the response with HTTP 500, and the
harness records an **empty completion** — which scores as an ordinary wrong answer. The
model looks mildly worse when it is actually failing to terminate.

**What it measures.** `build_eog_emit_map.py` accumulates per-layer routing weight at
positions predicting `<|im_end|>` (248046) / `<|endoftext|>` (248044), plus a background
arm over all other positions, then

```
lift[L][e] = emit-position weight share / background weight share
```

ranked **within** each layer, so a globally-hot expert cannot top the list merely by
being hot everywhere. `eog_keepset_gate.py` scores each arm's keep set by mean
within-layer lift rank and compares against a reference arm.

Build the map on the **same corpus** as the competence map, or the two are not
commensurable.

```bash
python runhost/build_eog_emit_map.py \
    --model <base> --corpus <router_calib_corpus_full.jsonl> \
    --eog-ids 248046,248044 --limit 0 --out results/eog_emit_map_ornith.json

python runhost/eog_keepset_gate.py \
    --eog-map results/eog_emit_map_ornith.json \
    --arm coder_tc=results/drop_map_184e_coder_tc.json \
    --arm coderx_p24=results/drop_map_184e_hybrid_p24.json \
    --reference coder_tc --max-drop 0.02
```

### Result on Ornith: INCONCLUSIVE — the replay does not cover the failure regime

```
mean EOG-lift rank01 over keep set (higher = retains more terminator mass)
  coder_tc                       0.4647    +0.0000
  coderx_p24                     0.4685    +0.0038
EOG_KEEPSET_GATE PASS
```

**Do not read this as "CoderX retains terminator mass."** The arithmetic is sound
(`EMIT_MASK_OK`: emit routings == emit_pos x K x L exactly), but the *corpus* is the
wrong replay for the defect it was built to investigate. Coverage audit of the 807-doc
router-calibration corpus:

| category | docs | tokens | truncated | emit (all) | emit (kept) | ends with EOG |
|---|---:|---:|---:|---:|---:|---:|
| targeted_lcb | 94 | 937,059 (84%) | **54** | 188 | **136** | **0** |
| targeted_mpe | 123 | 43,828 | 0 | 246 | 246 | 0 |
| gpqa_diamond | 80 | 29,189 | 0 | 160 | 160 | 0 |
| *(7 short cats)* | 510 | 109,172 | 0 | 940 | 940 | 0 |
| **TOTAL** | **807** | **1,119,248** | **54** | **1,534** | **1,482** | **0** |

Three compounding defects:

1. **Truncation cost emit positions.** `--max-tokens 4096` keeps the tail, so the *last*
   terminator per doc survives — but `targeted_lcb` still lost 52 of 188 (28%), and only
   479,128 of 1,119,248 corpus tokens (43%) were seen at all.
2. **The long-CoT regime is 9% of the map.** Every doc yields ~2 emit positions
   regardless of length, so `targeted_lcb` — 84% of the corpus by tokens — contributes
   136/1,482 positions. The map is **91% short-document turn boundaries.**
3. **No document ends with an EOG token** (`endsEOG = 0`, every category). The captured
   positions are *internal turn boundaries*, not "a long generation completed and the
   model chose to stop."

The observed defect appears after 13k+ tokens of thinking. This map measures short-context
turn boundaries. It therefore cannot confirm or refute EOG depletion in the failing
regime, and the PASS above is not evidence of termination health.

**What a valid EOG replay requires** (not yet built):

- documents that *end* the way the model ends a real generation — full CoT followed by the
  terminator — so `endsEOG` is non-zero;
- complete capture of each such document, windowed rather than truncated, so no emit
  position is discarded;
- emit positions dominated by the long-generation regime, not by short-doc turn
  boundaries; report the per-category census alongside the score.

The earlier Qwen3.6 cross-check (2026-08-20) that also returned negative was built on the
same corpus shape and inherits the same limitation.

## Open defect — CoderX empty completions (2026-09-07, UNRESOLVED)

Ornith CoderX Q6_K returns **20/77 empty completions** on `lcb_v6_77q` via llama-server
HTTP 500 (`common_chat_peg_parse: unparsed peg-native output`). A controlled comparison
on the identical binary, chat template, flags and problems:

| | Coder | CoderX |
|---|---|---|
| empty completions | **0 / 5** | **4 / 6** |
| PEG-500s | **0** | **4** |

Ruled out by measurement, all with Coder/base as the control:

- **chat template** — renders byte-identical single-turn; the fixed template
  (`templates/ornith_chat_template_fixed.jinja`, no forced `<think>` open) does not fix it
- **`--reasoning-format`** — fails under `deepseek`, under the `auto` default, and with
  the a3b serve config verbatim
- **llama.cpp version** — same binary (0.4.0-dev, 5266f24) ran Coder and base with zero failures
- **tokenizer / vocab / token types / prompt token counts** — identical across arms
- **EOG terminator depletion** — refuted above
- **runaway length** — Coder also hits the 32k ceiling (78k–104k chars) and still parses

Both arms overrun on hard problems; only CoderX's overruns fail to parse. Cause is in the
CoderX weights and remains unidentified.

## Known build defects

1. **`omk_ream_merge.py` strips the tokenizer.** `tok.save_pretrained()` rewrites
   `tokenizer_config.json` (16,718 B → 1,098 B), dropping `chat_template`,
   `added_tokens_decoder` (33 → 0), `additional_special_tokens`, `add_bos_token` and
   `extra_special_tokens`, and omitting `vocab.json` / `merges.txt`. With the embedded
   template gone, `convert_hf_to_gguf` falls back to the standalone `chat_template.jinja`
   — a *different* file that upstream also ships. Repair by copying the tokenizer from
   the base before converting (the a3b publish path does this). Affects CoderX only;
   Coder is built by the expert-drop path and is unaffected.
2. **Success-on-failure sentinels.** `p5_armB.sh` printed `ARM_B_EVAL_DONE` after its
   MPE cell died at warmup (`FATAL exit=20`). Sentinels must be gated on the artifact.
