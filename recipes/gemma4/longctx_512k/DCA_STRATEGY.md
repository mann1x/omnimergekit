# T87 — DCA strategy: train-to-256k, extrapolate to 512k–1M (Qwen2.5-1M lessons)

**Status:** strategic pivot proposed 2026-06-09, gated on a vLLM/llama.cpp DCA-for-Gemma4
feasibility investigation (consult in flight). Supersedes the "train at 512k" framing in
`TRAINING_PLAN.md` §6 as the *primary* path; that LoRA curriculum becomes the fallback.

## The goal (revised, user 2026-06-09)

Reach **faithful 1M context** on **both** Gemma 4 26B-A4B (MoE) and Gemma 4 31B (dense),
matching **Qwen2.5-14B-Instruct-1M**-class RULER/retrieval — which is still the strongest
open-weight "find the needle in 1M tokens" model despite being older and smaller than newer
MoE systems. **Long-context training matters more than raw parameter count for RULER-style
retrieval.** That is the bet.

### The motivating evidence — RULER @ 1M (the target curve)
| Model | RULER@128k | RULER@1M | 128k→1M drop |
|---|---|---|---|
| **Qwen2.5-14B-Instruct-1M** | **95.7** | **92.5** | **−3.2** |
| GPT-4o-mini | 87.3 | 74.5 | −12.8 |
| Llama-3-8B-Gradient-1M | 88.3 | 70.4 | −17.9 |
| Claude 3 (200K) | 90.1 | 73.2 | −16.9 |
| DeepSeek-V3 (128K) | 86.5 | 69.1 | −17.4 |

Two takeaways: (1) the **~3pt 128k→1M drop** is the signature of DCA+YaRN being *near-lossless
for retrieval* when the base is genuinely 256k-trained — the competitors fall 13–18pt because
they lack the training+DCA combo; (2) a **14B dense** model at 1M beats everything bigger/newer
by ~18–23pt. 31B dense is our direct analog; A4B is the MoE stretch. (Source: ajithp.com summary
of the Qwen2.5-1M report.)

## What Qwen2.5-1M actually did (arXiv:2501.15383)

The decisive fact: **they never trained at 1M.** Longest trained length = 256k. The final
4× (256k→1M) is **training-free at inference** via DCA + YaRN attention-scaling.

### 1. Progressive 5-stage length curriculum + RoPE-base ramp (ABF)
| Stage | Train len | RoPE base θ |
|---|---|---|
| 1 | 4k   | 10k → 1M (ABF) |
| 2 | 32k  | 1M |
| 3 | 64k  | 1M |
| 4 | 128k | 5M |
| 5 | 256k | 10M |

- Stages 3–5 data mix: **75% at current-max length + 25% shorter** (no short-ctx regression).
- **Adaptive Base Frequency (ABF):** raise RoPE θ with length (1M→5M→10M). This is a *different*
  lever than YaRN position-stretch — we currently use neither in the curriculum (fixed θ=1e6 +
  YaRN factor-2). Worth an A/B.

### 2. The curriculum validates itself (their Table 2, RULER on 14B)
Training at length N lifts *shorter* lengths too:
| Trained @ | RULER@64k | RULER@128k |
|---|---|---|
| 32k  | 76.4 | 37.6 |
| 64k  | 86.7 | 56.0 |
| 128k | 93.0 | 83.8 |
| 256k | 91.9 | 87.6 |

This is precisely the cliff-then-recover pattern our discriminator is probing. It's a real,
documented phenomenon and **longer training is the fix.**

### 3. Training-free length extrapolation (the part we want)
- **DCA (Dual Chunk Attention, An et al. 2024a):** split the sequence into chunks and *remap
  relative positions* so no query–key distance ever exceeds the pre-training length. Three
  regimes:
  - **intra-chunk** — tokens in the same chunk keep their native relative positions (fully
    trained range);
  - **inter-chunk** — clamped/repeated positions so cross-chunk distance ≤ train length;
  - **successive-chunk** — preserve local-window continuity between adjacent chunks, else
    fall back to inter-chunk clamping.
  Pure position-index transform → composes with FlashAttention; integrated in **vLLM** for Qwen.
- **YaRN attention-scaling (mscale only):** temperature on logits, √(1/t)=0.1·ln(s)+1,
  s = infer_len/train_len. Always paired with DCA. Neither alters short-context behavior.

### 4. Honest caveat — DCA amplifies, it does not substitute (their Table 4)
Qwen-7B trained@32k + DCA+YaRN → RULER@128k = **55.1**; the 256k-trained 1M model → **84.4**.
You still need a model trained reasonably long; DCA carries the last 2–4× beyond that.

### 5. Data lesson — natural text under-teaches long-range
Qwen: natural packed text has *weak* long-range dependencies (next-token is locally
predictable), so models never learn tracking. They add **synthetic long-range tasks**:
FIM, keyword/position retrieval, paragraph reordering. **VT (variable tracking) is exactly the
long-range axis failing for us, and `data_98e` is packed natural docs** — this likely explains
part of our VT cliff independent of DCA.

### 6. Inference cost (for serving 1M later, not training)
MInference sparse "Vertical-Slash" prefill + chunked-prefill + sparsity refinement → 3–7×
prefill speedup at 1M. Engine-side, not a training concern, but it's how 1M serving stays
affordable. vLLM-integrated.

## Mapping onto Gemma 4 (the crux)

- Gemma 4 is **natively 256k** (`max_position_embeddings=262144`). We need only **2×** to reach
  512k and **4×** to reach 1M — i.e. we are in the *same or easier* regime as Qwen's 256k-trained
  model, which extrapolated 4× cleanly.
- DCA need only act on the **5 global full-attention layers** (indices 5,11,17,23,29). The 25
  sliding-window layers (head_dim 256, window 1024) already bound relative distance by their
  window → DCA is a no-op there. This fits Gemma 4's hybrid structure cleanly.
- **Direction-of-effect evidence already in hand:** Qwen-7B trained@32k at 2× beyond train (64k)
  with DCA+YaRN ≈ 82 RULER-avg; our stage-1 (also 32k-trained LoRA) at 64k with **YaRN-only =
  0.74 VT**. Different scales, same message: DCA's position-clamping directly attacks the
  "untrained large relative distance" mechanism that *is* our cliff.

## The blocker — DCA is a serving-stack feature, and ours doesn't have it

- **vLLM** has DCA but wired for **Qwen** (Qwen2 attention path). Open questions: is the DCA hook
  model-generic or Qwen-coupled? Does it tolerate Gemma 4's **hybrid attention** (sliding+global
  interleave) and **head_dim=512** global layers (no FlashAttention kernel for hd>256 — upstream
  FA#2427 / our "bug-436")? DCA-with-FA assumes an FA kernel exists; on the hd=512 global layers
  we may need DCA over a non-FA SDPA path.
- **llama.cpp** has YaRN but **no DCA at all.** Either implement the position-remapping for the
  5 global layers ourselves, or stay on vLLM for the DCA path.

This is the make-or-break: **we must make DCA work for Gemma 4 in at least one of vLLM or
llama.cpp.**

## FEASIBILITY VERDICT — CORRECTED 2026-06-10 (council was WRONG)

⚠️ The council (csl-2026-06-09-2331-ea27) said "FEASIBLE — vLLM-only, ~2–3 days." **That is
wrong.** It verified the DCA *RoPE op* + model-level *wiring* exist, but never checked the
*attention backend* — where the 5× query is actually combined with the intra/inter/successive
chunk masks. **That executor was DELETED:** the only DCA backend vLLM ever had was the v0
`vllm/attention/backends/dual_chunk_flash_attn.py` (1495 lines), removed in `a815d820e` / #25351
"Remove V0 attention backends" (collateral of the #25321 V0-core deprecation — no DCA-specific
rationale; the RoPE op was kept). **No model — not even Qwen2.5-1M — runs DCA on vLLM v1 today.**
The remaining `dual_chunk_rope.py` + `get_rope` branch + `verify_dual_chunk_attention_config` + 9
Qwen-model wirings are vestigial. The v0 backend was also eager-only (`--enforce-eager`, dynamic
graph), env-gated, and partly broken pre-removal (PR #19084 short-seq, #19420 FP8).

**Real cost: a multi-week vLLM-v1 backend port** — reimplement the chunk-attention cleanly on a
CUDA-graph-friendly v1 Triton path (do NOT transcribe the v0 brute-force), AND make it run at
Gemma 4's head_dim=512 (FA caps at 256). DECISION (user, 2026-06-10): commit to the vLLM-v1 port.
Full plan: archived at `backup_models/docs/plans/plan_t87_dca_vllm_v1_port.md` (was
`/root/.claude/plans/cozy-bubbling-kernighan.md`); branch `feat/gemma4-dca`. The RoPE
half (5× producer + Gemma4-proportional hybrid + mscale) and the gemma4.py wiring below are still
correct and reusable; only the "easy/days" framing was wrong.

### Council revision 2026-06-10 (csl-2026-06-10-0608-a4a0) — three corrections, two verified in-tree
The corrected-status follow-up returned ~5–7 engineer-weeks and **fixed three things in our own plan**
(claims checked against `/shared/dev/vllm`@`630492da3`):
1. **Stage-1 reference is WRONG as written.** The plan said "compute DCA's 5-variant via plain
   `torch.scaled_dot_product_attention`." Plain SDPA over the 5×-**concatenated** query computes one
   softmax over the wrong (concatenated-head) axis — it does NOT merge five separate distributions.
   The correct reference is **5 separate attention calls → online-softmax / LSE merge**. vLLM already
   ships the exact primitive: `vllm/v1/attention/ops/merge_attn_states.py` (**verified**; already
   imported+used by the FA v1 backend for cascade/prefix attention at `flash_attn.py:989,1236`). S1
   must mirror that, not concatenate-then-SDPA.
2. **Pillar B softens — hd=512 is NOT categorically un-FlashAttention-able.** `flash_attn.py:181-188`
   `supports_head_size`: ≤256 always; **257–512 supported iff FA v4 is present** (`is_fa_version_supported(4)`;
   council said "FA3" — it's FA4). So if bs2's FA build is v4, the cheapest backend is **forking the FA
   path** (5× `flash_attn_varlen_func` + `merge_attn_states`) and skipping the custom Triton kernel;
   `triton_attn.py` (head_size≥32) is the fallback when FA4 is absent. **Check bs2 FA version at Stage 0.**
   This may cut the estimate.
3. **NEW highest-risk unknown: `softcap=50.0` × LSE merge.** Gemma 4 applies tanh logit soft-capping on
   the global layers. Each of the 5 per-variant attention calls applies softcap to its own logits; the
   LSE merge must be mathematically consistent with softcap-over-full-attention, or the output diverges.
   Make this the **explicit Stage-1 gate assertion** (softcapped 5-way LSE merge ≈ softcapped full attn).
   Second risk: DCA's `positions % chunk_len` remap vs Gemma 4's proportional/partial-rotary RoPE.
- CUDA-graph escape from the v0 `--enforce-eager`: pre-allocate a `max_chunks_per_sequence` buffer
  (≈4 for 1M) + Triton-mask actual chunk counts → `AttentionCGSupport.ALWAYS`. Paged-KV unaffected
  (the position remap doesn't touch block table / slot mapping). Community v1-DCA effort: **zero**.

### (Council's per-file map — still useful for the RoPE/wiring half; verify line #s on synced HEAD)

- **DCA RoPE op is generic & hd=512-safe.** `DualChunkRotaryEmbedding` (`dual_chunk_rope.py:14`,
  `forward_native:122-184`) is pure PyTorch, FA-agnostic. `Gemma4Config.verify_and_update_config`
  (`config.py:58-107`) already forces **TRITON_ATTN** when global_head_dim>256, so the no-FA-kernel
  issue (FA#2427) is already handled — DCA's tensors ride that path fine.
- **Gemma 4 has ZERO DCA wiring** (`gemma4.py`), unlike the Qwen family. Adding it is the work.
- **CATCH 1 — DCA vs proportional RoPE are mutually exclusive in `get_rope()`** (`__init__.py:86-100`):
  setting `dual_chunk_attention_config` bypasses `Gemma4RotaryEmbedding` (the proportional/partial-
  rotary class, `gemma4_rope.py:16-84`) entirely. So this is NOT a config flip — we must build a
  **hybrid `Gemma4DualChunkRotaryEmbedding`** merging Gemma4's proportional `inv_freq`
  (`gemma4_rope.py:40-42`) with DCA's position remap.
- **CATCH 2 — DCA emits a 5× query tensor** (q, q_succ, q_inter, q_succ_critical, q_inter_critical).
  `Gemma4Attention.forward` (`gemma4.py:502-536`) must handle it. **At hd=512 with no FA kernel,
  this 5× expansion on the SDPA/Triton path is the real PERFORMANCE risk** — one lane called it a
  "production blocker" (effective head_dim≈2560/query). Feasibility is clear; *1M throughput* is the
  open question — likely needs MInference-style sparse attention too (attention = ~90% of forward at
  1M per the Qwen report). Affects only 5/30 layers, but they're the expensive global ones.
- **YaRN mscale is NOT in Gemma 4's proportional path** — add it as a separate attention-logit
  temperature (scalar on `self.scaling`, `gemma4.py:402`). mscale = 0.1·ln(s)+1 → **1.0693 @512k,
  1.1386 @1M**. (Note: **llama.cpp already exposes `yarn_attn_factor`** — `llama-arch.cpp:261`,
  `llama.h:347-351` — so the YaRN/mscale half is free there; only DCA is missing in llama.cpp.)
- **MoE routing-collapse risk = LOW.** `Gemma4Router` (`gemma4.py:252-298`) reads hidden states
  (residual stream), not position IDs; DCA's remap touches attention only. 31B dense = zero risk by
  construction. Validate empirically on 31B first, then A4B with the existing `routing_entropy_probe`.

### Minimal vLLM path (council's 5 steps)
1. Patch `Gemma4Attention.__init__` (`gemma4.py:482-500`) to accept `dual_chunk_attention_config`,
   thread to `get_rope()` + `Attention()`.
2. Apply **only** when `self.is_full_attention` (`gemma4.py:558`) → layers 5,11,17,23,29.
3. Build `Gemma4DualChunkRotaryEmbedding` = proportional inv_freq ⊕ DCA position remap.
4. Update `Gemma4Attention.forward` for the 5× query tensor.
5. Add mscale as an attention-temperature scalar (outside RoPE).
6. Config: `dual_chunk_attention_config = {"chunk_size": 262144, "local_size": 1024}`.

### First experiment (cheap, decisive)
Wire DCA into vLLM gemma4 → serve the **31B BASE (untrained)** at 512k → RULER VT/NIAH. This tests
the Qwen thesis directly: does native-256k + DCA+YaRN extrapolate 2× training-free? If yes, we've
reproduced the Qwen result on Gemma 4 with near-zero training. If partial, add the light 256k
continued-pretrain (hybrid path). Then A4B with routing-entropy monitoring; then push to 1M.

## Revised plan (primary = DCA, fallback = LoRA curriculum)

1. **PRIMARY — DCA+YaRN extrapolation (training-free or near-free).**
   Exploit Gemma 4's native 256k; chunk to 512k then 1M; tune YaRN mscale. Gated on the DCA
   feasibility verdict. If feasible, this collapses the 4×RTX6000 + Ulysses + E1/E2/E3 training
   engineering into a serving-stack feature.
2. **Hybrid (likely best): light continued-pretrain to 256k under the target rope, THEN DCA.**
   Qwen's caveat says extrapolation needs a long-trained base. If our native-256k + YaRN base
   isn't a strong enough "256k-trained" anchor, a short LoRA pass to re-assert 256k under the
   serving rope (+ ABF) precedes DCA. Far cheaper than training at 512k.
3. **FALLBACK — staged LoRA curriculum** (`TRAINING_PLAN.md` §6), improved by Qwen's two free
   wins regardless: **(a)** add synthetic long-range tasks (FIM / retrieval / reorder) to the
   calib corpus; **(b)** 75/25 current-max/shorter length mix; **(c)** ABF θ-ramp A/B vs YaRN.
   And **stop at 256k + extrapolate**, never train at 512k.

## Relation to the running discriminator (disc2_64k, ~08:00 CEST 2026-06-10)

Still worth finishing: it answers "does LoRA-at-length work" (the fallback lever) and
independently confirms the cliff mechanism Qwen documents. But it is no longer the project's
critical path — the DCA feasibility verdict is. If the discriminator confirms LENGTH-WALL, that
*supports* the hybrid path (training does move the needle at length); it does not argue against
DCA.

## Open questions for the feasibility consult
1. vLLM DCA: implementation locus; model-generic vs Qwen-coupled; Gemma 4 hybrid + hd=512 support.
2. llama.cpp DCA: any PR/branch; effort to add position-remap for the 5 global layers only.
3. Does the transformers/vLLM rope path for Gemma 4 even expose YaRN **mscale** (prior open
   item csl-2026-05-28-1825-5f1b)? DCA's YaRN half depends on it.
4. Minimal end-to-end path to a DCA+YaRN Gemma 4 serving 512k, then 1M, on bs2 (2×RTX6000) /
   4×RTX6000 rental.
5. Does DCA compose with Gemma 4 **MoE routing** at >256k (A4B) — any routing-collapse risk the
   position remap could trigger? (31B dense avoids this; A4B does not.)
