# JackOD-9B-Coder — recipe

The full, reproducible chain behind
[`ManniX-ITA/JackOD-9B-Coder`](https://huggingface.co/ManniX-ITA/JackOD-9B-Coder)
and its quants,
[`ManniX-ITA/JackOD-9B-Coder-MTP-GGUF`](https://huggingface.co/ManniX-ITA/JackOD-9B-Coder-MTP-GGUF).

The build arm is named `JackOD4` in the scripts — it was the fourth weighting
tried. The published model is that arm; the codename is kept in the scripts so
the logs and this recipe line up, and is dropped from the release name.

## What it is

A four-way `omnimerge_v2` merge over a single shared ancestor, aimed at
**multi-turn agentic coding**: keeping a task moving to completion across turns
and tool calls, rather than maximising single-shot benchmark scores.

| role | model | weight |
|---|---|---|
| base / task-base | `Qwen/Qwen3.5-9B` | — (shared ancestor) |
| source | `DeltaCoder-9B-applied` (LoRA on `Qwen/Qwen3.5-9B`, merged to dense) | 0.55 |
| source | `Jackrong/Qwopus3.5-9B-coder` (base `Jackrong/Qwopus3.5-9B-v3.5`) | 0.30 |
| source | `Ornith-1.5-9B` | 0.15 |

`--base` and `--task-base` are both `Qwen3.5-9B`: every source descends from it,
so the task vectors are taken against the true common ancestor rather than
against one of the siblings.

## 1. Merge

```bash
python omnimergekit.py \
  --base      models/Qwen3.5-9B \
  --task-base models/Qwen3.5-9B \
  --source    DeltaCoder-9B-applied \
  --source    models/Qwopus3.5-9B-Coder \
  --source    models/Ornith-1.5-9B \
  --weights   0.55,0.30,0.15 \
  --method omnimerge_v2 --density 0.53 --darex-q 0.75 --seed 42 \
  --no-auto-mlp-skip --skip-patterns visual.,mtp. \
  --output JackOD4-9B-Coder
```

`--skip-patterns visual.,mtp.` leaves the **vision tower (333 tensors)** and the
**MTP head (15 tensors)** untouched, passed through from the base. Merging them
is not a neutral act: an MTP head that gets averaged across sources stops being
a usable drafter, and a `save_pretrained` round-trip can drop it entirely — the
shipped model is checked for both (`census_jackod.py`).

See [`build_jackod4.sh`](build_jackod4.sh).

## 2. Root-EOS patch — not optional

`patch_jackod.py` copies the root `eos_token_id`, tokenizer and chat template
from Qwopus onto the merge.

**Why:** `Qwen3.5-9B` and `DeltaCoder-9B-applied` carry **no root
`eos_token_id`** — it lives only in the nested text config. The GGUF converter
reads the root, finds nothing, and falls back, which puts the wrong terminator
in the quant and leaks control tokens into generations. The merge serves
Qwopus's chat template, so it takes Qwopus's terminator with it.

## 3. Calibration — the AC corpus

```bash
bash bs2_jackod4_ac_imatrix.sh     # full-corpus imatrix, 9686 chunks, --parse-special
```

The imatrix is built on the **AC corpus** (`calib_train.txt`, 17.6 MB) over
**all** chunks, not the 128-chunk default — 128 chunks is ~65k tokens and would
consume ~1.3% of the corpus.

**An imatrix is not transferable between merges.** It is a fingerprint of one
specific set of weights: a different weighting, a different expert cut or a
different source set needs its own. The published imatrix
(`imatrix.dat`, sha256 `02d25d65…`) is archived in the GGUF repo so every quant
can be audited and reproduced.

## 4. Quantize + publish

```bash
bash bs2_jackod_quant_sweep.sh     # all tiers -> HF, imatrix on every _K/IQ tier
bash bs2_upload_jackod_q6k.sh      # the pre-built Q6_K + the imatrix
```

`Q6_K` is excluded from the sweep and uploaded separately: it already existed,
built from the same F16 with the same AC imatrix, and it is the artifact every
tool-calling number in the model card was measured on. Rebuilding it would have
produced a second file that no published measurement refers to.

Published GGUF filenames follow the **repo** name
(`JackOD-9B-Coder-<TIER>.gguf`), never the build-time arm codename.

## Serving

Greedy is the right default for reproducible measurement. For use, the shipped
`generation_config.json` carries the serving recipe:

```
temperature 0.6 · top_p 0.95 · top_k 20
```

plus `presence_penalty 1.5` where the runtime supports it (ollama does; HF
`GenerationConfig` has no such field). The penalty is what keeps multi-turn
agentic runs from re-treading the same tool call.
