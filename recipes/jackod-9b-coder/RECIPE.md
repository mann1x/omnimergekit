# JackOD-9B-Coder — recipe

The full, reproducible chain behind
[`ManniX-ITA/JackOD-9B-Coder`](https://huggingface.co/ManniX-ITA/JackOD-9B-Coder)
and its quants,
[`ManniX-ITA/JackOD-9B-Coder-MTP-GGUF`](https://huggingface.co/ManniX-ITA/JackOD-9B-Coder-MTP-GGUF).

**The published weights are v2** (re-released 2026-10-21, in place of v1, same
repo). v2 is the v1 recipe with one change: Jackrong's **Coder-Flash** in the
0.30 slot that held Qwopus3.5-9B-Coder. Everything else — method, density,
darex-q, seed, weights, skip patterns, base — is identical. The v1 chain is kept
below as the record of how the recipe was chosen.

**JackOD** is an initialism of the three sources: **Jack** = JackRong's Qwopus
9B Coder (v2: Coder-Flash), **O** = Ornith-1.5-9B, **D** = DeltaCoder.
`Qwen3.5-9B` is absent from the name because it is the shared ancestor the merge
is taken *against*, not a flavour contributed to the blend.

## What it is

A four-way `omnimerge_v2` merge over a single shared ancestor, aimed at
**multi-turn agentic coding**: keeping a task moving to completion across turns
and tool calls, rather than maximising single-shot benchmark scores.

| role | model (pinned revision) | weight |
|---|---|---|
| base / task-base | `Qwen/Qwen3.5-9B` @ `c2022362` | — (shared ancestor) |
| source | `danielcherubini/Qwen3.5-DeltaCoder-9B` @ `10a2546f` (root LoRA r32/a32, applied to dense) | 0.55 |
| source | `Jackrong/Qwopus3-5-9B-Coder-Flash` @ `a972c214` (v1: `Jackrong/Qwopus3.5-9B-coder`) | 0.30 |
| source | `ornith-ai/Ornith-1.5-9B` @ `489cb979` | 0.15 |

`--base` and `--task-base` are both `Qwen3.5-9B`: every source descends from it,
so the task vectors are taken against the true common ancestor rather than
against one of the siblings.

## 1. Merge (v2)

```bash
WORK=/path/to/workdir OMK=/path/to/omnimergekit bash build_jackod_v2.sh
```

[`build_jackod_v2.sh`](build_jackod_v2.sh) downloads the pinned inputs, applies
the DeltaCoder LoRA to dense with `scripts/apply_lora_to_safetensors.py` (never a
transformers load/save round-trip, which drops `mtp.*` and nests key prefixes),
and runs:

```bash
python omnimergekit.py \
  --base      models/Qwen3.5-9B \
  --task-base models/Qwen3.5-9B \
  --source    DeltaCoder-9B-applied \
  --source    models/Qwopus3-5-9B-Coder-Flash \
  --source    models/Ornith-1.5-9B \
  --weights   0.55,0.30,0.15 \
  --method omnimerge_v2 --density 0.53 --darex-q 0.75 --seed 42 \
  --no-auto-mlp-skip --skip-patterns visual.,mtp. \
  --output JackOD-9B-Coder
```

`--skip-patterns visual.,mtp.` passes the **MTP head (15 tensors)** through from
the base untouched: an MTP head averaged across sources stops being a usable
drafter. The `visual.` half is a no-op here — base and merge both carry
**775 tensors and zero `visual.*`**; the `vision_config` in `config.json` is
inherited architecture metadata with no weights behind it, and the model is
text-only. `census_jackod.py` gates every stage on exactly the base's tensor set
with the MTP head present.

## 2. Root-EOS patch — not optional

`Qwen3.5-9B`, `DeltaCoder-9B-applied` **and Coder-Flash** carry **no root
`eos_token_id`** — it lives only in the nested text config. The GGUF converter
reads the root, finds nothing, and falls back, which puts the wrong terminator
in the quant and leaks control tokens into generations.

v1 copied the root eos from Qwopus (`patch_jackod.py`). Coder-Flash has none to
copy, so v2 takes the tokenizer and chat template from Coder-Flash
(`patch_jackod.py`) and **forces** root `eos_token_id` 248046 / `pad_token_id`
248055 — Qwopus' values, the same ones v1 shipped. The v2 chat template differs
from v1's; the tokenizer is identical.

## 3. Calibration — the AC corpus (v2)

```bash
MODEL=... OUT=... AC=.../calib_train.txt LC=.../llama.cpp-311d4211bf OMK=... bash bs2_jackod_v2_ac_imatrix.sh
```

The imatrix is built on the **AtomicChat** corpus
(`AtomicChat/calib-corpora`, `builds/ornith-1.5-9b/calib_train.txt`, sha256
`a47fdfee…`) over **all 9686 chunks** with `--parse-special`, on llama.cpp
`311d4211bf` — not the 128-chunk default, which would consume ~1.3% of the
corpus.

**An imatrix is not transferable between merges.** It is a fingerprint of one
specific set of weights: v1's `02d25d65…` does not apply to v2, which has its
own, archived as `imatrix.dat` in the GGUF repo so every quant can be audited and
reproduced.

## 4. Quantize + publish (v2)

```bash
MODEL=.../JackOD-9B-Coder OUT=... IMATRIX_SHA=<sha of OUT/imatrix.dat> LC=... OMK=... bash bs2_jackod_v2_sweep.sh
```

[`bs2_jackod_v2_sweep.sh`](bs2_jackod_v2_sweep.sh) builds `Q6_K` first (the tier
the model card's measurements use), then v1's tier set — the bartowski ladder
plus the legacy CD ladder including the opt-in `CD-Q2_K` — with the imatrix on
every `_K`/IQ tier, gated on the imatrix sha before and after. The weights dir
must be named `JackOD-9B-Coder`: published filenames and `general.name` follow
the **repo** name, never a build-time codename.

## Serving

Greedy is the right default for reproducible measurement. For use, the shipped
`generation_config.json` carries the serving recipe — see the model card for the
current values and the reasoning behind them (temperature, repetition window).

---

## v1 (previous weights, replaced 2026-10-21)

The v1 build arm is named `JackOD4` in its scripts — it was the fourth weighting
tried. It used `Jackrong/Qwopus3.5-9B-coder` (base `Jackrong/Qwopus3.5-9B-v3.5`)
in the 0.30 slot.

| step | script |
|---|---|
| merge | [`build_jackod4.sh`](build_jackod4.sh) |
| root-eos patch (eos + tokenizer + template from Qwopus) | [`patch_jackod.py`](patch_jackod.py) |
| full-AC imatrix (`02d25d65…`) | [`bs2_jackod4_ac_imatrix.sh`](bs2_jackod4_ac_imatrix.sh) |
| Q6_K on that imatrix | [`bs2_jackod4_ac_quant.sh`](bs2_jackod4_ac_quant.sh) |
| tier sweep (Q6_K excluded) + Q6_K upload | [`bs2_jackod_quant_sweep.sh`](bs2_jackod_quant_sweep.sh), [`bs2_upload_jackod_q6k.sh`](bs2_upload_jackod_q6k.sh) |

The v1 `Q6_K` was excluded from its sweep and uploaded separately because it was
the artifact every v1 tool-calling number was measured on; rebuilding it would
have produced a second file no published measurement refers to.
