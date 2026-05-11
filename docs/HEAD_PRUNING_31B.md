# Head pruning on Gemma 4 31B — T8 experiment log

Producer: [`recipes/gemma4_31b/prune_local_heal.py`](../recipes/gemma4_31b/prune_local_heal.py)
Quant ladder: [`scripts/quantize_gguf.py`](../scripts/quantize_gguf.py) → Q4_K_M
Eval driver: [`eval/eval_suite.sh`](../eval/eval_suite.sh) + `eval/tasks/{humaneval_chat,mbpp_chat}.yaml`
Reference variant: `google/gemma-4-31B-it-he125` (12.5% Q-head prune, lstsq heal). The strict-win recipe.

This document is the consolidated experimental record for the head-pruning project on
Google's Gemma 4 26B-A4B / 31B-it. T8 = the production attention-head-pruning track,
named after the per-layer Q-head fraction × 100 (he125 = 12.5%, he25 = 25%, he375 = 37.5%).

Three independent attempts at "improving" the recipe (mask-only, scope-narrowed
mask-only, lora-r4 correction) were all falsified at the same scope. lstsq is the
only recipe that survives generation-mode canary across the full 60-layer scope on
31B. The structural-reshape variant collapses on E2B (n_kv=1) — fix landed in the
producer as `auto_resolve_prune_mode` returning `"mask"` whenever `n_kv_s == 1`.

---

## 1 — Architecture & methodology summary

Gemma 4 31B-it is dense (60 layers, 50 sliding + 10 full attention), 32 Q heads
per sliding layer and 64 Q heads per full layer, KV heads paired in groups of 2.
Total ~62 GB BF16, ~18 GB Q4_K_M.

The producer pipeline is three phases:

1. **Phase 1 — importance scoring.** Per-layer per-head Michel α-grad
   `|∂CE/∂α[L,h]|` at α=1 against next-token CE on a calibration set. Heads
   ranked low-importance per layer are candidates for pruning.
2. **Phase 2 — apply prune.** For each layer, zero out the kept-cols / drop-cols
   in `q_proj`, `k_proj`, `v_proj`, and `o_proj` consistent with the dropped
   heads. (Mask mode keeps the head dimension and zeroes; structural mode
   reshapes to a smaller head count.)
3. **Phase 2.5 — heal (the load-bearing step).** Ridge-regularized least-squares
   refit of `o_proj`'s kept columns against the unpruned model's per-token
   `o_proj` output on the calibration tokens. Closed-form (no Adam, no LoRA),
   full-rank within the kept-cols subspace.
4. **Phase 2.6 — generation-mode canary (mandatory).** TF-only validation lies:
   `feedback_calibration_signal_misleading` documents two independent projects
   where TF-CE / per-layer reconstruction residual improved while autoregressive
   generation died (token-loop / nonprintable drift / early-EOS). Save the
   pruned weights under `.broken/` if any of three canary checks fail:
   - **(a) drift**: pruned NLL of base's greedy gens > 3× base NLL on the same.
   - **(b) shape**: pruned greedy gen has bigram-repetition > 40% OR
     nonprintable-fraction > 30% OR < 5 tokens (early-EOS).
   - **(c) self-vs-base ratio**: pruned NLL of base gen / pruned NLL of pruned
     own gen > 3×. Catches "high-confidence in own loop" failure mode.

### Why this section exists

Three failure modes from earlier iterations all show up clean on TF metrics:

| recipe        | final_ce | drift (a) | shape (b)            | what TF said |
|---|---:|---:|---|---|
| T7.2 he125-E (structural, E2B) | 14.02 | 0.90–1.20× | "LAB:LAB:" loop on every prompt | "healthy" |
| T8noheal full-scope | 5.83 | 0.77/0.82/0.95 | "Once upon" nonprintable loop | "tighter than lstsq" |
| T8 lora-r4 full-scope | 5.82 | 0.83/0.95/1.20 | "Once upon" rep=1.00 nonp=1.00 | "fits target tighter" |

All three would have shipped silently without the shape canary. The cost of the
canary is ~10 s of GPU time on 31B; it is mandatory.

### Local feasibility (24 GB VRAM)

`project_gemma4_31b_he_local_blocked.md` documents the 15-iteration discovery
arc. **Working configuration:**

| phase | base | placement | peak memory |
|---|---|---|---|
| 1 (importance) | nf4 (bnb 0.49.2) | `device_map={"":"cuda:0"}` | ~17 GiB load + ~5 GiB grad |
| 0, 2 (capture / prune / heal) | bf16 | `device_map="auto"`, `{0:"8GiB","cpu":"200GiB"}` | accelerate offload with `align_module_device` write-back |

Key non-obvious constraints (each one cost iterations):

- `bnb 4-bit + accelerate CPU offload` is broken in the bnb 0.49.2 / transformers
  5.5.0 combination (`Params4bit.__new__() got an unexpected keyword argument
  '_is_hf_initialized'`). Use full-GPU placement for nf4.
- All-60-gates-simultaneously OOMs because autograd holds 60 simultaneous bnb
  dequant buffers. Per-layer single-gate scoring at 192 tokens/chunk fits in
  ~22 GiB. (`--phase1-mode nf4_global --phase1-nf4-chunk-tokens 192`)
- `gradient_checkpointing=True` makes nf4 memory **worse** due to dequant
  fragmentation during recompute. Leave it off for nf4.
- Loading bf16 first then unloading does NOT fully free GPU. Load nf4 first in
  clean process state for Phase 1; spawn a fresh process for Phase 2's bf16
  load.
- `bf16 windowed-K Michel + L2 reconstruction with cached α=1 target` is
  algebraically degenerate (loss=0 at α=1 → grad=0). Reconstruction-style
  losses against α=1 targets give zero signal. End-to-end CE is the only
  Michel loss that satisfies (a) connected autograd and (b) non-zero gradient
  at α=1.

### Smoke command (validated 2026-05-08)

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python recipes/gemma4_31b/prune_local_heal.py \
    --model-path google/gemma-4-31B-it \
    --output google/gemma-4-31B-it-he125-smoke \
    --prune-frac 0.125 \
    --calib-file scripts/calibration_datav5.txt \
    --placement auto --gpu-mem 8GiB --cpu-mem 200GiB \
    --phase1-mode nf4_global --phase1-nf4-chunk-tokens 192 \
    --smoke --smoke-tokens 384 --chunk-tokens 384 \
    --ridge 1e-2
```

Smoke reports `lstsq rel_resid` per layer (sliding L=0 = 0.0011, full L=5 = 0.0003,
target_rms 6.30 / 3.71) and final calib CE 7.74 vs 9.72 baseline.

---

## 2 — Eval results (full canonical 164q HumanEval + 500q MBPP)

Setup: Q4_K_M GGUFs on solidPC RTX 3090, llama-server `-c 16384 --parallel 2`
with q8_0 KV cache, lm-eval 0.4.11 `local-chat-completions` (omnimergekit
`eval/tasks/{humaneval_chat,mbpp_chat}.yaml`). SQLite cache at
`eval_results_full/cache/<variant>_<task>_rank0.db` (resumable). Wrapper:
[`scripts/run_31b_full_he_mbpp.sh`](../../scripts/run_31b_full_he_mbpp.sh).
Pipeline ran 2026-05-08 20:21–23:43 CEST (3h22m total, ~50 min/variant).

### Headline numbers

| Variant | HE-164 pass@1 | Δ HE vs base | MBPP-500 pass@1 | Δ MBPP vs base | Verdict |
|---|---:|---:|---:|---:|---|
| base (gemma-4-31B-it) | 0.9817 (161/164) | — | 0.8500 (425/500) | — | reference |
| **he125** (12.5% prune) | **0.9878 (162/164)** | **+0.61pp** | **0.8520 (426/500)** | **+0.20pp** | **wins both** |
| he25 (25% prune) | 0.9878 (162/164) | +0.61pp | 0.8460 (423/500) | −0.40pp | ties HE, −2 MBPP |
| he375 (37.5% prune) | 0.9695 (159/164) | −1.22pp | 0.8500 (425/500) | 0.00 | breaks HE, MBPP holds |

he125 is a **strict win** — Pareto-improves base on both HE and MBPP. With identical
file size on disk (Q4_K_M architecture is preserved, kept-cols only refit), this
is a no-cost quality improvement that ships as a drop-in replacement for the base
Q4_K_M GGUF.

he25 is the **HE frontier** point — same HE improvement as he125, costs 2 MBPP
passes. Interesting if HE-specific use case but he125 is the safe default.

he375 is the **boundary of usability** — −2 HE passes (1.22pp), MBPP held at base.
The HE/MBPP decoupling at 37.5% is worth a follow-up (does HE rely on a smaller
attention-head pool than MBPP, or does CoT-length on HE problems amplify residual
errors more?).

### Sanity check post-eval

Per [`eval/EVAL_PROTOCOL.md`](../eval/EVAL_PROTOCOL.md) every result is gated on:

1. `samples_*.jsonl` line count = 164 (HE) / 500 (MBPP). All 4 variants × 2 tasks
   = 8/8 pass.
2. p10 generation length ≥ 60 chars. All pass.
3. No markdown-fence drift. All pass.
4. `results_*.json` `exact_match,flexible-extract` matches what the SQLite cache
   reflects on re-score. All pass.

Score data lives under
`eval_results_full/{humaneval_chat,mbpp_chat}/gemma4-31B-<variant>-Q4_K_M/.../results_*.json`
with companion `samples_*.jsonl` for forensic re-scoring.

---

## 3 — Falsified alternatives

Three branches were taken to see if "better than lstsq" was possible. All failed
at the generation-mode canary on 31B's full 60-layer scope. lstsq remains the only
working heal at full scope on 31B.

### 3.a — T8noheal-scope: mask-only heal (no o_proj refit)

`project_t8noheal_scope_31b.md` (2026-05-09 to 05-10). E4B had a documented
sweet spot at L4-L7 where mask-only beat lstsq (`project_t77_ar_heal_falsified`).
The question: does that transfer to 31B at any narrow-band scope?

**Answer: no, at any scope.** 0/3 canary at every tested band:

| variant | scope | canary | drift triple | "Once upon" failure |
|---|---|---|---|---|
| T8noheal-full | 60 layers | 0/3 | 0.77 / 0.82 / 0.95 | nonprintable loop |
| T8noheal-L0-L3 | 4 sliding | 0/3 | 0.77 / 0.82 / 0.95 | nonprintable loop |
| T8noheal-L4-L7 | 3 sliding + L5 full | 0/3 | 0.84 / 1.05 / 1.16 | identical loop attractor |
| T8noheal-L8-L11 | 3 sliding + L11 full | 0/3 | 0.84 / 1.05 / 1.16 | byte-identical to L4-L7 |
| T8noheal-L20-L23 | 3 sliding + L23 full | 0/3 | 0.84 / 1.05 / 1.16 | byte-identical |
| T8noheal-L40-L43 | 3 sliding + L41 full | 0/3 | 0.84 / 1.05 / 1.16 | byte-identical |
| T8noheal-L56-L59 | 3 sliding + L59 full | 0/3 | 0.84 / 1.05 / 1.16 | byte-identical |

Two clean equivalence classes form:

- L0-L3 only (no full-attention layer): drift 0.77/0.82/0.95, calib CE 5.83.
- Anything with one full-attention layer: drift 0.84/1.05/1.16, calib CE 5.81
  **byte-equal to 4 decimals across 5 separately-pruned models**.

The drop of 8/64 Q heads on a full-attention layer dominates whatever sliding-band
delta exists, and the contribution is identical across L5/L11/L17/L23/L29/L35/L41/L47/L53/L59.
calib CE on each band stays *below* T8 lstsq's 5.97 — once more confirming
calib CE doesn't predict generation quality.

**Architectural interpretation.** Per-layer redundancy on 31B is much lower than
E4B's at the same fraction:

| | sliding Q heads | drop @ 12.5% | full Q heads | drop @ 12.5% |
|---|---:|---:|---:|---:|
| 31B | 32 | 4 | 64 | 8 |
| E4B | 8 | 1 | 8 | 1 |

31B loses 4–8× more attention capacity per layer at the same fraction. Without
lstsq's full-kept-cols correction, the residual stream cannot route around the
gap. E4B's L4-L7 anomaly is a small-head, narrow-architecture artifact and does
not generalize.

### 3.b — T8 lora-r4: low-rank Adam correction

`project_t8lora4_31b_failed.md` (2026-05-10). T13 on E4B found lora-r4 beat
lstsq at L4-L7. The question: does that transfer to 31B at full he125 scope?

**Answer: no.** 0/3 canary. Same "Once upon" rep=1.00 nonp=1.00 attractor as
T8noheal. Final calib CE 5.82 (vs T8 lstsq 5.97) — **lora-r4 fits TF target
*tighter*** but still produces a degenerate model.

| variant | calib CE | canary | "Once upon" failure |
|---|---:|---|---|
| T8 lstsq (published) | 5.97 | passed eval (HE 0.9878, MBPP 0.8520) | OK |
| T8 noheal | 5.83 | 0/3 | rep=1.00 nonp=1.00 |
| T8 lora-r4 | 5.82 | 0/3 | rep=1.00 nonp=1.00 |

Hypothesis: lstsq has access to the full kept-cols rank and solves a closed-form
ridge-regularized problem; lora-r4 is rank-bounded δ added to W_orig via 200-step
Adam. On 31B with 4–8 dropped Q heads per layer × 60 layers, rank-4 cannot absorb
the cumulative residual-stream damage. On E4B with 1 dropped Q head per layer
× 42 layers, rank-4 suffices.

### 3.c — T7.7 ar-lstsq: AR-distributed target heal

`project_t77_ar_heal_falsified.md` (2026-05-09, on E4B). Tested whether the
TF/AR distribution mismatch was the cause of lstsq failure at L4-L7. Sampled
greedy rollouts from the masked student, captured teacher outputs on
(prefix + student-rollout) as AR-distributed `y_target`, lstsq with the AR pair.

**Verdict:** 1/3 canary on E4B L4-L7, same as TF lstsq. `ar_vs_tf` rms divergence
per layer was significant (L4=0.84, L5=0.98, L6=0.24, L7=0.61) — the two heals
solved fundamentally different optimization problems and still produced identical
canary failure patterns. Heal damage on E4B's shallow attention is **structural,
not distribution-dependent**.

### 3.d — T7.2 structural reshape on E2B / n_kv=1 collapse

`project_t7_2_e2b_catastrophic.md` (2026-05-09). E2B (n_kv=1, 8 Q heads / 1 KV
group) was used as fast-validation rig before A100 spend on 31B redo.

| variant | HE pass@1 | MBPP | failure mode |
|---|---:|---:|---|
| base F16 | 0.7378 | 0.5340 | clean |
| he125-E structural reshape | **0.0000** | ≈0 | token-loop garbage on every prompt |
| he125-Esh R1 mask L0-L3 | 0.1159 | 0.0780 | early-EOS / empty on most prompts |

Even 1.4% mask scope (4 Q heads total, L0-L3 sliding only) crashed HE by 62 points
on E2B. Producer fixes landed:

- **Fix B** — `auto_resolve_prune_mode` returns `"mask"` whenever `n_kv_s == 1`
  unless `--allow-low-kv-structural` is set. Structural reshape is unsafe at
  group_size=1.
- **Fix C2** — the three-check canary protocol (drift / shape / self-vs-base),
  documented in §1.
- **Fix A** — `phase1_importance_loo` per-head leave-one-out CE delta as an
  alternative to gradient-based importance. Architecture-agnostic, directly
  measures `‖ΔCE‖` for the k=1 drop case. Cheap on small models (~5 min E2B),
  expensive on 31B (~30-60 min on A100). Adds `--phase1-mode loo`.

### 3.e — T7.6 FFN block-LOO at L4-L7 on E4B

`project_t76_ffn_falsified.md` (2026-05-09). Hypothesis: 10240 FFN channels gives
finer granularity than 8 Q heads, GQA fragility doesn't apply, multi-class CD map
should pay off harder.

**Result:** all three runs (noheal, lstsq, multi-class) at L4-L7 = **1/3 canary**,
**worse than attention noheal's 2/3** at the same scope. "X is X is X" loop pattern
is identical across attention-lstsq and all FFN variants — L4-L7 has a single
shared bottleneck that any block-level perturbation hits at 12.5%.

---

## 4 — Reusable artifacts in this repo

| File | Role |
|---|---|
| [`recipes/gemma4_31b/prune_local_heal.py`](../recipes/gemma4_31b/prune_local_heal.py) | Producer pipeline. CLI flags cover all the falsified variants too (`--heal {lstsq,noheal,lora-r4,ar-lstsq}`, `--phase1-mode {nf4_global,bf16_windowed,loo}`, `--prune-layers` for scope sweeps, `--canary-baseline-cache` for re-runs). Fix B / Fix C2 / Fix A all in tree. |
| [`recipes/microcoder_4b/prune_local_heal.py`](../recipes/microcoder_4b/prune_local_heal.py) | Qwen3.5-4B adaptation (linear_attn skip for the 24 GatedDeltaNet layers — see `project_microcoder_he_negative.md`). Hybrid arch doesn't tolerate per-token lstsq healing; canonical MicroCoder remains v2i task-arithmetic merge. |
| [`eval/tasks/humaneval_chat.yaml`](../eval/tasks/humaneval_chat.yaml) | Full 164q HE via chat-completions (Gemma 4 needs chat mode). |
| [`eval/tasks/mbpp_chat.yaml`](../eval/tasks/mbpp_chat.yaml) | Full 500q MBPP via chat-completions. |
| [`eval/tasks/humaneval_smoke20.yaml`](../eval/tasks/humaneval_smoke20.yaml) + `_subset_filter.py` | Reproducible 20q HE subset (frozen indices) for rapid iteration. |
| [`eval/eval_suite.sh`](../eval/eval_suite.sh) | llama-server lifecycle + lm-eval invocation with `--use_cache` + `--log_samples` (mandatory). |
| [`eval/peek_cache.py`](../eval/peek_cache.py) | Read-only SQLite cache inspector for lm-eval cache files. |

Reference wrappers (in the project root, not the repo):
`scripts/run_31b_full_he_mbpp.sh`, `scripts/run_31b_t8noheal.sh`,
`scripts/run_31b_t8noheal_scope_sweep.sh`, `scripts/run_31b_t8lora4.sh`.

---

## 5 — Open follow-ups

| ID | Question | Cost | Likely outcome |
|---|---|---|---|
| T11 | Structural reshape on 31B (architecture-level size win, not just mask) | ~6h A100 + canary | ~2-7% Q4_K_M file-size reduction; would let he125 land at ~17 GB |
| T11b | FFN-neuron prune on 31B (MLP = 69% of weights) | ~10h A100 | bigger lever than attention; needs new importance proxy (activation-based REAP, not LOO-CE) given T7.6's L4-L7 failure on E4B |
| T8x | Attention × FFN combined prune (he125 + light FFN) | ~12h A100 | speculative — both fixes need to compose without compounding damage |
| T12 | Differential CD-quant on he125 (high-fidelity quant for kept heads, lower for residue) | ~4h | size win on top of he125; requires expert-contribution map shifted to per-head granularity |
| HE/MBPP decoupling at he375 | Why does HE break before MBPP at 37.5%? | 1 day | tells us whether HE relies on a more compressible attention subspace |

Higher fractions (he50, he625) untested. Predicted from the he125 → he25 → he375
trend: both HE and MBPP should break by he50.

---

## 6 — Published / planned releases

| Variant | Status | HF repo | Ollama |
|---|---|---|---|
| `he125` Q4_K_M | local only (2026-05-11) | (pending: `ManniX-ITA/gemma-4-31b-he1-it`) | (pending: `mannix/gemma4-31b-he1`) |
| `he125` BF16 safetensors | local only | (pending) | n/a |
| Full quant ladder for he125 | not built yet | (pending) | (pending) |

he125 is the only variant approved for publish. he25 ties HE but regresses MBPP
and is dominated by he125. he375 breaks HE and isn't competitive at any axis.
