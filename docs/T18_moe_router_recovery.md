# T18 — MoE router recovery for post-prune Gemma 4

**Goal**: fix the rumination + silent-empty pathology that 98e expert-pruned
Gemma 4 26B-A4B exhibits relative to the 128e baseline. The drop step
preserves expert weights but leaves the **router pointing to dropped slots**,
which under top-8 softmax-over-remaining-98 produces a noisy off-manifold
routing on hard prompts → the model loops, dumps reasoning into thinking
tokens until the budget cap, then truncates with no usable answer.

This document describes the **Step 0 / 1 / 2 / 3 framework** built up over
T17 (drop-map) + T18 (router recovery). Scripts live in
[`recipes/gemma4/v5_moe_sweep/`](../recipes/gemma4/v5_moe_sweep/).

## The pathology

Two failure modes observed on 98e variants vs 128e baseline:

1. **Rumination** — the same prompt that 128e answers in 538 chars takes
   23,000 chars on 98e; nearly all of those chars are repeated reasoning
   bullets, prompt-echoes, multiple aborted code attempts.
2. **Silent-empty** — the chat-completion endpoint returns `content=""` for
   many problems; reasoning is consumed but never surfaced. Multiple causes:
   (a) Gemma 4 reasoning parser closure-capture bug (vLLM #42250, since
   fixed), (b) thinking-budget exhaustion that swallows the actual answer,
   (c) router instability that hangs the forward pass.

Both modes correlate with **router instability**. The 128e router learned a
top-8 softmax over all 128 slots; after `expert_drop.py` removes 30 rows,
the surviving 98 are renormalised but the routing mass that used to flow to
the dropped slots is now arbitrarily redistributed by chance (the softmax
denominator changes, but the row weights themselves are identical to the
128e router). On the calibration distribution the 128e was trained on, the
top-8 over 98 is close to the top-8 over 128 of the same prompts. On
**held-out hard prompts** (HumanEval / GPQA / MBPP), it isn't — the dropped
slots used to absorb a meaningful fraction of mass for those distributions.

## Step 0 — audit current router state

Goal: quantify the routing-distribution drift between 128e and 98e on the
calibration vs the held-out evals. Identify which layers are most affected.

* **Renormalisation audit**: confirm `router.proj.weight` rows in the 98e
  NVFP4A16 dir are bit-identical to the corresponding 128e surviving rows
  (modulo NVFP4A16 quantisation). `expert_drop.py` slices but does not
  retrain.
* **Top-k entropy probe**: forward-pass a small calibration set
  through both 128e BF16 and 98e NVFP4A16; capture
  `pre_feedforward_layernorm` outputs; compute top-8 softmax + entropy
  per layer. Layers with high entropy on 98e but low on 128e are the
  routing-damaged layers.
* **Silent-empty probe**:
  [`probe_router_silentempty.py`](../recipes/gemma4/v5_moe_sweep/probe_router_silentempty.py)
  — runs a small targeted prompt set through the 98e via vLLM, logs
  empty-content responses, and dumps per-layer routing entropy for the
  silent cases. Output:
  [`router_probe_silentempty.json`](../recipes/gemma4/v5_moe_sweep/router_probe_silentempty.json)
  (captured 2026-05-10 for v4; still relevant signal for v5).

## Step 1 — free-knob recovery

Three reversible NVFP4A16-dir edits that don't require re-quantisation
(important: re-quantising every variant under tuning takes ~5 min each on
solidpc — 100× sweep would be 8h. Reversible edits land in seconds).

### Step 1a — `router_topk_dial.py`

The config.json carries `top_k_experts: 8` (the count of experts kept after
softmax). At 98 surviving experts, top-8 over a smaller denominator means
each kept expert receives a higher post-softmax weight than in the 128e
case. Dropping to `top_k_experts: 6` reduces routing density to roughly
match the 128e effective mass (~6/128 ≈ 0.047 per kept slot, vs ~6/98 ≈ 0.061
post-prune).

```bash
python router_topk_dial.py --model-dir <NVFP4A16> --top-k 6
python router_topk_dial.py --model-dir <NVFP4A16> --restore  # undo
```

Reversible (atomic config.json swap; `top_k_experts` key tracked).

### Step 1b — `router_shared_upweight.py`

Gemma 4 has a parallel always-on dense `mlp.*` FFN at every MoE layer that
runs alongside the routed mixture. After expert drop, the routed mixture is
unreliable on hard prompts; leaning more on the **shared** FFN (α > 1) masks
the routing damage.

The `mlp.*` weights are NVFP4A16 (4-bit). We do NOT touch the 4-bit
weights — we scale the FP32 `weight_scale_2` of `mlp.down_proj` by α. Since
dequantised weight = `(qweight × weight_scale) × weight_scale_2`,
multiplying `weight_scale_2` by α multiplies the layer's shared FFN output
by α. Same numerical effect as re-quantising at α-scaled weights, without
the re-quant cost.

```bash
python router_shared_upweight.py --model-dir <NVFP4A16> --alpha 1.2
python router_shared_upweight.py --model-dir <NVFP4A16> --restore
```

Reversible (per-shard `.pre_shared_upweight` backups).

### Step 1c — `router_soft_transfer.py`

Distribute each dropped expert's router-mass onto its cosine-similar
surviving cousins. Inverse of "Diversifying Expert Knowledge" merge (ACL
Findings 2025): for each dropped expert d in the BASE 128e router, find
the top-k cosine-similar surviving expert rows in the same base router. Add
a weighted fraction of d's row to those survivors' rows in the VARIANT
router. At inference, hidden states that would have lit up d now (partially)
excite its closest cousins instead.

```bash
python router_soft_transfer.py \
    --base-dir google/gemma-4-26B-A4B-it \
    --variant-dir <NVFP4A16> \
    --drop-map <drop_map.json> \
    --alpha 0.3 --top-k 3
python router_soft_transfer.py --base-dir ... --variant-dir ... --drop-map ... --restore
```

Reversible (per-shard `.pre_soft_transfer` backups).

### Composite — `apply_t18_step1.sh`

Orchestrator that runs all three Step 1 transforms with chosen knobs, then
re-runs the same smoke triplet as the sweep (gsm8k_30 + HE-1 + LCB-1).
Appends results to `logs/t18_step1_summary.tsv`. Each transform is
`--restore`able so multiple knob settings can be A/B'd on the same NVFP4A16
dir.

```bash
bash apply_t18_step1.sh <variant_tag> [topk] [alpha_shared] [alpha_transfer]
# defaults: topk=6 alpha_shared=1.2 alpha_transfer=0.3
```

## Step 2 — EAC-MoE TopK-MSE calibration

Reference: arXiv:2508.01625 (EAC-MoE: Expert-and-Calibration co-tuning for
pruned MoE LLMs). Idea: keep the routing identity but **retrain the
`router.proj.weight` rows** on a small calibration corpus to match the
teacher's top-K positions, with the dropped positions explicitly masked.

`router_eac_calibrate.py` (in `recipes/gemma4/v5_moe_sweep/`) is a full
implementation in two phases:

**Capture phase**:
* Hook `pre_feedforward_layernorm` on the base 128e BF16 model
* Forward-pass the calibration corpus (WikiText-2 by default, or
  `--corpus-file <path>` for task-specific)
* Stream per-layer hidden states to `eac_cache/h_layer_NN_batch_MM.pt`

**Calibrate phase**:
* Per layer, AdamW-optimise `router.proj.weight` (BF16) to match teacher's
  top-K positions, with the dropped positions explicitly zeroed in the loss
* Loss = MSE between student's softmax and teacher's softmax over the
  surviving 98 positions (top-K positions from the 128e teacher, masked to
  the 98 surviving experts via the drop map)
* Optimise for `--steps` (default 200) per layer; learning rate
  `--lr` (default 1e-4)
* Saves the updated router weights back into the variant's NVFP4A16
  safetensors (BF16 stays BF16)

```bash
# A/B run: WikiText-2 baseline vs task-specific differential corpus
python router_eac_calibrate.py --phase both --base-dir google/gemma-4-26B-A4B-it \
    --variant-dir <NVFP4A16-A>

python router_eac_calibrate.py --phase both --base-dir google/gemma-4-26B-A4B-it \
    --variant-dir <NVFP4A16-B> \
    --corpus-file logs/diff_corpus.txt
```

### Differential calibration corpus (DiffCal)

Original user concept: pick HF eval samples where 128e gets the answer
RIGHT but is NOT ruminating (response length below the bottom-N% per
bench). The signal: "this is what correct routing on this task looks like".

`build_diff_corpus.py` reads `samples_*.jsonl` from
`eval_results_vllm_suite/128e/<bench>/...` for HumanEval, gsm8k_100,
math500_100, ifeval_100, ifeval_full, humanevalplus_full. Filters:
1. **Correct**: `pass@1==1.0` or `exact_match==1.0` per bench's metric
2. **Non-ruminative**: response length ≤ 70th percentile of correct
   responses for that bench

Result 2026-05-16: **423 examples / 103,915 tokens**:

| Bench | Total | Correct | Kept (≤70th pct len) | Tokens |
|-------|------:|--------:|---------------------:|-------:|
| HumanEval | 164 | 163 | 115 | 34,839 |
| HumanEval+ | 164 | 148 | 104 | 30,394 |
| gsm8k_100 | 200 | 190 | 134 | 22,448 |
| math500_100 | 100 | 5 | 4 | 1,012 |
| ifeval_100 | 100 | 90 | 63 | 14,127 |
| ifeval_full | 5 | 4 | 3 | 1,095 |

Renders each (prompt, response) pair through Gemma 4's chat template,
concatenates with `<|im_end|>\n\n` separator. router_eac_calibrate.py
consumes it as plain text (re-tokenised at capture time by the same
tokenizer).

```bash
python build_diff_corpus.py \
    --root eval_results_vllm_suite/128e \
    --tokenizer google/gemma-4-26B-A4B-it \
    --out logs/diff_corpus.txt \
    --rumination-percentile 70 \
    --max-examples 0
```

A/B comparison plan: run EAC twice per candidate (once with WikiText-2,
once with `--corpus-file logs/diff_corpus.txt`). Score post-EAC variants on
the same smoke triplet. Compare deltas.

## Step 3 — Full Router KD

Reference: Hyeon & Do, arXiv:2603.02217 ("Router Knowledge Distillation for
Post-Hoc MoE Pruning"). Full KD from the 128e teacher's routing
distribution; ~hours per variant vs Step 2's ~15 min. Only run if Step 2
deltas are insufficient (< +5pp on the target benches).

Not yet built — pending Step 2 results.

## Reproduction — full T18 chain on one variant

```bash
TAG=B4_max_broad

# Step 1 (all three knobs at defaults)
bash apply_t18_step1.sh $TAG 6 1.2 0.3
# → eval_results_vllm_suite/v5fixed_t18/${TAG}__t6_s1.2_x0.3/

# Step 2A — WikiText-2 baseline
python router_eac_calibrate.py --phase both \
    --base-dir google/gemma-4-26B-A4B-it \
    --variant-dir google/gemma-4-A4B-98e-v5fixed-sweep-${TAG}-NVFP4A16

# Step 2B — task-specific DiffCal
python router_eac_calibrate.py --phase both \
    --base-dir google/gemma-4-26B-A4B-it \
    --variant-dir google/gemma-4-A4B-98e-v5fixed-sweep-${TAG}-NVFP4A16 \
    --corpus-file logs/diff_corpus.txt

# Compare via re-smoke
bash apply_t18_step1.sh $TAG  # re-runs the smoke triplet post-Step-2
```

## Status — 2026-05-16

* **Scripts: READY** (Step 1 + Step 2)
* **Diff corpus: BUILT** (423 examples / 103,915 tokens)
* **Sweep: COMPLETE** (T19, 13 variants) — top candidates identified:
  B4 (all-rounder), B2 (math), B3 (code).
* **Pending**: longer-smoke validation on top-3 + actual Step 1/2 application.

## References

* [arXiv:2510.13999](https://arxiv.org/abs/2510.13999) — REAP (Router
  recalibration baseline that Step 1 sits on top of)
* [arXiv:2508.01625](https://arxiv.org/abs/2508.01625) — EAC-MoE TopK-MSE
  (Step 2)
* [arXiv:2603.02217](https://arxiv.org/abs/2603.02217) — Hyeon & Do Router KD
  (Step 3 reference)
* [ACL Findings 2025: Diversifying Expert Knowledge](https://aclanthology.org/2025.findings-acl.) —
  cosine-similar merge inverse used in Step 1c
* [memory/reference_moe_router_recovery_methods.md] (solidpc-local) —
  expanded survey
