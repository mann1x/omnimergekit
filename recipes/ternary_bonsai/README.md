# Ternary recovery in the Bonsai 2 format — recipe and tests (closed, negative result)

> **Status: closed 2026-10-14.** We did not get close to PrismML's own ternary models
> and stopped. This page records what we built and measured, and where a restart
> would begin. Nothing from this line was published to Hugging Face. The packing tool
> (`tools/opencoti-quantize/`, byte-identical to PrismML's) and the opt-in `PQ2_0` /
> `PTQ1_0` tiers of `quantization/quantize_gguf.py` are separate and stay supported.

**Goal.** Ternarize our own models (Omnimerge v6, a Qwen3.8-27B merge) into the format
of PrismML's Ternary Bonsai 2: the same GGUF types, the same runtime, and coding quality
close to Bonsai 2.

**Outcome.** Our best 27B ternary scores **43.3 %** on HumanEval. Its teacher scores
79.3 %, and Bonsai 2 scores 70.1 %. At 4B the gap is wider: our best gets **17.1 %**,
PrismML's ternary 4B gets 66.5 %.

## The format we target

We reverse-engineered it from PrismML's files (`bonsai_scale_rule.py`,
`bonsai_layer_lab.py`, `gguf_read_prism_meta.py`):

- every linear layer of the Bonsai contract is folded into a Hadamard-rotated basis
  (`bonsai_hadamard_fold.py`);
- in that basis, each group of 128 weights along the input axis keeps its top 86
  magnitudes as ±1 and zeroes the other 42 (zero fraction 0.328);
- each group has one scale;
- the weights are packed as `PQ2_0` (2-bit) or `PTQ1_0` (1.6-bit);
- `ssm_alpha` and `ssm_beta` stay bf16, and so do the norms and embeddings.

The packing is lossless against the unpacked ternary weights. On the 27B wiki text, the
bf16 file holding exactly ±d gives KL 0.522546 and the packed `PQ2_0` gives 0.522708.
Our `PQ2_0` / `PTQ1_0` files match PrismML's tensor by tensor in type and shape.

## What we tried, and the numbers

**How KL is measured:**

- **Trainer KL** is computed by the trainer itself: 32 × 1024-token windows, top-64 plus
  a tail term (`--eval-teacher online` gives the full KL). The eval texts are
  `wiki.test`, plus held-out teacher conversations (code / tech / general / agent)
  where noted.
- **llama.cpp KL** is `llama-perplexity --kl-divergence` on the PrismML llama.cpp fork,
  `-c 2048`.
- Every ternary model is measured against **its own** teacher.

**How HumanEval is run:** lm-eval `humaneval` through `local-completions`, greedy, the
same invocation for every row (`bonsai_he_refs.sh`). Every sample file was checked for
empty, fenced and junk generations before its score was used.

### 1. Training-free ternarization, 27B (`bonsai_v6_ternarize_sweep.sh`, `bonsai_base_vs_prism.sh`)

| Ternarizer (teacher Qwen3.8-27B) | wiki KL | same top token |
|---|---|---|
| magnitude top-86 (the strict format, no training) | 8.24 | 5.0 % |
| top-86, scale 0.9 | 9.77 | 1.1 % |
| TWN, threshold 0.75 | 4.49 | 20.7 % |
| L2-optimal scale | 4.99 | 14.9 % |
| **PrismML Ternary Bonsai 2** | **0.376** | **78.2 %** |

**Conclusion:** no rounding rule comes near. Bonsai 2 is a trained model, not a rounding
of its base.

### 2. Error-compensating PTQ, 27B (PT2-LLM style: GPTQ + iterative ternary fit + activation-aware grid)

Scripts: `bonsai_pt2_full.py`, `bonsai_pt2_export.sh`, `bonsai_pt2_combined.sh`,
`bonsai_pt2_selftrace.sh`, `bonsai_trace_domain_kld.sh`, `bonsai27_he_chain.sh`.

| Arm | wiki KL | held-out-trace KL | HumanEval |
|---|---|---|---|
| mixed calibration corpus | 1.206 | 0.564 | — |
| mixed corpus + teacher's own traces | — | 0.307 | 27.4 % |
| teacher's own traces only (AYOT) | — | 0.301 | — |
| PrismML Bonsai 2 | 0.376 | 0.210 | 70.1 % |

### 3. Ternary recovery training (QAT with a frozen teacher)

Scripts: `bonsai17_qat.py` (1.7B), `bonsai4b_qat.py` (4B, AdamW), `bonsai27_qat.py` (any
size up to 27B on one 96 GB GPU). All three share `bonsai_tern_lib.py`, which
`bonsai_tern_lib_test.py` tests on CPU.

The method:

- latent weights live in the rotated basis;
- the forward pass serves top-86 of the latent with a straight-through estimator;
- the loss is distillation on the teacher's cached top-64 logits;
- the learning rate warms up, then follows a cosine decay to zero.

Every run first passes a **lossless gate**: with ternarization switched off, the
student must reproduce its teacher (KL ≤ 0.001).

**Qwen3-1.7B (teacher Qwen3-1.7B, FineWeb-Edu stream), trainer wiki KL:**

| Arm | Steps | wiki KL | Note |
|---|---|---|---|
| r1 magnitude start | 2000 | 0.905 | strict format |
| r1 GPTQ start | 2000 | 0.759 | not the strict format (45 % zeros) |
| r3 GPTQ start under keep-86 | 2000 | 0.859 | strict format, −5 % vs magnitude |
| r2 magnitude start | 8000 | 0.658 | |
| r2 GPTQ start | 8000 | 0.626 | |
| r4 25 % / 50 % teacher-trace mix | 2000 | 0.916 / 1.001 | trace KL 0.257 / 0.222 (control 1.200) |
| community reproduction (alllwang) | — | 0.568 | HumanEval 5.5 % |
| **PrismML Ternary 1.7B** | — | **0.454** | HumanEval 47.0 % (base 40.2 %) |

On a neutral held-out set (GSM8K conversations, `bonsai17_r5_chain.sh`), the 25 % trace
mix reaches 0.563 against the control's 0.817. PrismML reaches 0.269.

**Qwen3.5-4B (teacher Qwen3.5-4B, HumanEval 59.8 %):**

| Arm | Steps | wiki | code | tech | general | agent | HumanEval |
|---|---|---|---|---|---|---|---|
| r1 control, FineWeb only (AdamW) | 2000 | 0.779 | 1.536 | 1.229 | 1.309 | 2.426 | 0.6 % |
| r1 25 % teacher-trace mix (AdamW) | 2000 | 0.800 | 0.732 | 0.382 | 0.723 | 0.507 | 10.4 % |
| `bonsai27_qat.py`, lr 2e-5 | 2000 | 0.779 | 0.647 | 0.364 | 0.669 | 0.554 | 17.1 % |
| same + AtomicChat corpus, ~5M tokens (`bonsai27_4b_ac.sh`) | 2606 | 0.783 | 0.630 | 0.326 | 0.674 | 0.417 | 17.1 % |
| **PrismML Ternary 4B** (from Qwen3-4B, base 70.7 %) | — | | | | | | **66.5 %** |

Adding the AtomicChat corpus lowered agent-text KL by 25 %. HumanEval did not move: the
same 28 of 164 problems pass in number, with 22 passing in both runs, 6 lost and 6 gained.
The AC run also had 2,606 steps against 2,000, so data and steps are confounded.

**Omnimerge v6 27B (teacher Omnimerge v6 bf16, `bonsai27_v6_chain.sh`, 2000 steps):**

- **Trainer KL** on wiki / code / tech / general / agent:
  0.493 / 0.288 / 0.195 / 0.494 / 0.508.
- **Packed `PQ2_0`, each model against its own teacher** (`bonsai27_v6_target_kld.sh`;
  Bonsai 2 against Qwen3.8-27B):

| Text | ours KL | Bonsai 2 KL | ours same top | Bonsai 2 same top |
|---|---|---|---|---|
| wiki | 0.523 | 0.376 | 73.2 % | 78.2 % |
| held-out traces | 0.261 | 0.210 | 87.9 % | 91.4 % |
| code | 0.293 | 0.123 | 86.8 % | 92.3 % |

| | HumanEval |
|---|---|
| Omnimerge v6 bf16 (teacher) | 79.3 % |
| PrismML Ternary Bonsai 2 27B | 70.1 % |
| **ours, r1_v6 `PQ2_0`** | **43.3 %** |

Paired against the teacher: 65 problems pass in both, 65 are lost and 6 are gained. In
the samples, 0 are empty, 4 are fenced and 1 is junk.

The fourth text, agent conversations, is void. There the teacher's own perplexity
(22.3, and 56.0 for Qwen3.8) is higher than its student's, so the text is out of domain
for both teachers.

## Why we stopped

- **The ternary pattern is fixed at initialisation.** Over the whole 27B run only 1.07 %
  of the zero/non-zero mask changed, and no weight changed sign. Training only adjusts
  scales and latents inside the pattern it started from. The calibrated (GPTQ) starts
  bought 5–16 % KL at 1.7B. Matching Bonsai 2 on code would need −58 %.
- **More steps do not help.** Every curve is flat over its last 250 steps, because the
  learning rate reaches zero. Even so, the 27B trainer code KL (0.288) already equals the
  llama.cpp figure for the packed file, so the measurements are consistent and nothing
  is left hidden.
- **More or better data moves domain KL, not code quality.** At 4B, teacher traces and
  the AtomicChat corpus both lower KL on their own domain. HumanEval stays at 17 %
  against PrismML's 66 %.

PrismML's gap is in the training regime itself: its length, data, schedule, and probably
how the pattern is chosen during training. It is not in the format, the packing or the
evaluation, which all reproduce.

## Restarting

The cheapest step that could move the result would change how the pattern is chosen,
not train longer:

- re-select the top-86 mask during training, which ours never does;
- or start from an error-compensating ternarization of the right shape.

What a restart needs:

- **Scripts:** all are in `scripts/` (index below). Hosts and paths are written into
  each script's header variables.
- **Teacher caches:** rebuild them with `--phase cache`. `bonsai27_ac_cache.py` contains
  the reproduction gate showing that a rebuild matches the original exactly (top-1
  agreement 1.0000).
- **Kept outside the repo:** every log, eval result and HumanEval sample, the generated
  replay corpora (1.7B / 4B / 27B), and the final trained models. These are the 27B
  `PQ2_0` file, sha `7ab4658b…`, and the dense exports of the four final 4B arms. The
  latent checkpoints and caches were not kept.

## Script index

| Script | Role |
|---|---|
| `bonsai_tern_lib.py`, `bonsai_tern_lib_test.py` | Shared ternary latent, fold and export pieces; CPU self-test |
| `bonsai_hadamard_fold.py` | Fold a qwen35 GGUF into the Bonsai 2 Hadamard basis (`--override-dir` writes trained trits) |
| `gguf_read_prism_meta.py` | Dump a GGUF's KV section without gguf-py (which rejects the PrismML types) |
| `bonsai_scale_rule.py`, `bonsai_layer_lab.py`, `bonsai_layer_gptq.py`, `bonsai_prefix_test.py`, `bonsai_capture_acts.py` | Single-layer forensics of PrismML's trits and scales on real activations |
| `bonsai_qat_pilot.py`, `bonsai_pt2_blocks.py`, `bonsai_pilot_run.sh`, `bonsai_pt2_run.sh` | First-L-blocks pilots: rotation-in-the-loop QAT, and PT2, with their runners |
| `bonsai_v6_ternarize_sweep.sh`, `bonsai_base_vs_prism.sh`, `bonsai_v6_fold_quant.sh`, `bonsai_v6_fold_quant_mtp.sh` | Training-free ternarizers on 27B and the like-for-like comparison with Bonsai 2 |
| `bonsai_rsprot_ablation.sh`, `bonsai_rsprot_prism.sh` | PrismML's one documented gen1→gen2 method change, measured |
| `bonsai_pt2_full.py`, `bonsai_pt2_export.sh`, `bonsai_pt2_combined.sh`, `bonsai_pt2_selftrace.sh`, `bonsai_pt2_mix_finish.sh` | Whole-model PT2-style ternarization of 27B, and its calibration-corpus arms |
| `bonsai_gen_self_traces.py`, `bonsai_trace_domain_kld.sh`, `bonsai_trace_spotcheck.py` | Teacher self-traces as calibration text; held-out-trace KL |
| `bonsai_corpus_plan.py`, `bonsai_gen_corpus_srv.py`, `bonsai_gen_corpus_srv2.py`, `bonsai_corpus_stats.py` | Replay-corpus plan, generation (vLLM) and completeness table |
| `bonsai17_*.py/.sh` | Qwen3-1.7B: baselines, trace generation and filtering, QAT trainer, rounds r2–r5. `bonsai17_qat.py` is the final trainer (identical to `.v4`). `bonsai17_qat.v1.py`–`.v4.py` are the exact files the rounds ran (v1 = baselines and r2, v2 = r3, v3 = r4, v4 = r5), kept because each round script names its own version |
| `bonsai4b_*.py/.sh` | Qwen3.5-4B: corpus, AdamW QAT trainer, export, round r1, HumanEval chain, sample sanity |
| `bonsai27_qat.py`, `bonsai27_export.py` | Memory-lean QAT trainer (27B on one 96 GB GPU) and dense export |
| `bonsai27_val4b_g1.sh`, `bonsai27_full4b.sh` | Validation of `bonsai27_qat.py` on 4B against the AdamW run, then the full 4B run |
| `bonsai27_corpus.sh`, `bonsai27_v6_chain.sh`, `bonsai27_he_chain.sh`, `bonsai27_v6_target_kld.sh` | Omnimerge v6 27B: replay corpus, end-to-end run, HumanEval, KL against its own teacher |
| `bonsai27_ac_cache.py`, `bonsai27_4b_ac.sh` | AtomicChat corpus added to a teacher cache; the 4B AC arm |
| `bonsai_he_refs.sh`, `bonsai_eval_table.py` | HumanEval yardstick for published ternary models; EVAL-log table |
| `bonsai_identity_probe*.py/.sh`, `bonsai_identity_show.py` | What a ternary model says about itself vs its base |
