# T87 — Gemma 4 26B-A4B long-context (→512k) training plan & audit brief

Status: **DRAFT for council audit.** The author (Claude) is presenting this with low
confidence; the user has explicitly asked for an independent, skeptical audit because
prior recommendations under-scoped the problem. Challenge everything below, including
the RCA.

## 0. What actually happened (the honest gap)

The goal is a YaRN-extended Gemma 4 26B-A4B that serves *faithfully* at long context
(trained config: `rope_type=proportional_yarn`, factor 2.0, `original_max_position_embeddings`
262144, text `max_position_embeddings` 524288 = 512k).

What was **executed and evaluated** is only **stage 1 of an implied staged curriculum**:
a single LoRA continued-pretraining run at **32k sequence length**.

- Adapter: `ckpt_98e_ddp32k_fa2/ckpt-000952`, LoRA **r16 / α32 / dropout 0**, targets
  **q_proj, k_proj, o_proj on the 5 global-attention layers only** (layers 5,11,17,23,29).
  No v_proj, no MLP. ~2.07 M trainable params.
- Data: 250 M tokens, packed at **32768**. `.merge_done` shows `packs_yielded 7624`,
  `seen_tokens 249823232` (= 7624 × 32768) → confirms 32k packs, not longer.
- Trainer: `scripts/phase1_train_yarn_lora.py`, DDP across 2 GPUs, grad_accum 4,
  ce_chunk 2048, grad-ckpt on, memeff attention. loss 5.88 → 1.30.

It was then merged (`gemma-4-26B-A4B-it-512k`), GGUF-converted (F16), and evaluated on
RULER as if it were the finished 512k product. **It is not** — it is the 32k stage of a
curriculum whose later stages were scaffolded but never run.

**Why this wasn't surfaced earlier (author's admission):** the work was scoped around the
eval/serving harness (does proportional⊕YaRN serve faithfully?), and the 32k checkpoint was
treated as a correct given input. The `ddp32k` in the dir name and the 32k pack count in
`.merge_done` were visible but not audited until the long-context eval failed.

## 1. The evidence — RULER VT (variable tracking) cliff, llama.cpp F16, single-slot

Same backend for ext and base. ext = proportional base rope (in GGUF) + YaRN at runtime
(`--rope-scaling yarn --yarn-orig-ctx 262144 --rope-scale 2.0 --yarn-beta-fast 32
--yarn-beta-slow 1`). base = stock proportional, no YaRN.

| ctx  | ext (trained+YaRN) | base (no YaRN) | \|Δ\| |
|------|--------------------|----------------|-------|
| 32k  | 0.92               | 0.948          | 0.028 |
| 64k  | 0.69 (approx)      | ~0.95          | ~0.26 |
| 128k | ~0.64              | ~0.97          | ~0.33 |
| 256k | 0.62               | 0.972          | 0.352 |

(64k/128k from the cliff sweep; 32k/256k are the anchors. Exact 64k/128k numbers on bs2 —
`/srv/ml/longctx/ruler_llama/cliff_*/`.) NIAH/retrieval (mk1) holds far better than VT;
**it is long-range *tracking/reasoning* that collapses, not retrieval.**

**Author's RCA:** the degradation tracks a **32k training-length wall** — the model holds
where it was trained (32k, Δ≈0.03) and degrades monotonically beyond it; base holds ≥0.95
throughout, so serving/rope is faithful and the deficit is *trained capability*, not a bug.
**This RCA is item #1 to audit.** Alternative hypotheses to rule in/out: (a) LoRA capacity
too small / wrong modules for tracking; (b) YaRN ramp params (beta_fast/slow) mistuned for
this base; (c) data lacks genuine long-range-dependency examples; (d) attention-only LoRA
cannot move a reasoning operation.

## 2. Hardware reality on bs2 (the constraint that drives everything)

bs2 = 2× RTX PRO 6000 Blackwell, **97887 MiB each**, 346 GB RAM. Harness has **no
ring/context/sequence parallelism** — only DDP (data-parallel, replicates the seq) and
model-parallel (`--gpus 0,1` → `device_map balanced_low_0`, splits *layers* not *sequence*).

Measured ceilings (`/srv/ml/longctx/*.log`):
- **64k seq, single GPU**, grad_accum 1: peak **95.6 / 97.9 GB** — barely fits one Blackwell.
- **64k seq, model-parallel both GPUs**: peak **63.5 GB** (per the post32k PP/MP smoke).
- Throughput collapses with length: 4640 tok/s (short) → **1686 tok/s at 64k**.
  250 M tokens @ 64k ≈ **41 h**. 128k is slower and multi-day.
- The trainer's own `--pack-len` help literally says **"32k..64k; NOT 256k"** and the 31B
  launcher path *forces* `pack_512_frac=0.0` because "31B can't fit 512k packed chunks on a
  single 96 GB Blackwell." The author of the harness already capped expectations at ~64k.

**Implication:** bs2 can train **up to ~64k, possibly ~128k under MP + grad-ckpt + activation
offload**, but **cannot train true 256k/512k sequences.** A genuine ≥256k training stage needs
either ring/context parallelism (not present) or cloud (H100/H200 multi-node).

## 3. Proposed staged-length curriculum (the plan to audit)

Resume the existing r16/α32 LoRA from `ckpt-000952` and extend the trained length in stages,
re-packing data at each length, with a NIAH gate between stages:

| Stage | pack-len | GPUs (mode)         | tokens | est. wall | gate (NIAH-256k) |
|-------|----------|---------------------|--------|-----------|------------------|
| 1 ✅  | 32768    | DDP ×2              | 250 M  | done      | n/a (smoke)      |
| 2     | 65536    | MP `--gpus 0,1`     | ~250 M | ~41 h     | ≥0.80, abort on 3× |
| 3     | 131072   | MP `--gpus 0,1` + offload | ~150–250 M | multi-day | ≥0.80 |
| 4 (512k) | 262144+ | **cloud only**   | tbd    | tbd       | tbd              |

Existing scaffolding to reuse (in this dir):
- `phase1_train_yarn_lora.py` — trainer (`--gpus` MP, `--ddp`, `--resume`, `--ckpt-every-steps`,
  `--offload-activations`, `--grad-ckpt`, `--max-mem-gib 92`).
- `pack_pg19_math_rpv2.py` — re-packer (pg19 + math + rpv2; `--pack-512-frac` to mix long docs).
- `phase1_probe_watcher.py` — per-ckpt NIAH-256k probe, `--abort-on 3 --threshold 0.80 --kill-pid`.
- `patch_yarn_config.py`, `proportional_yarn_rope_init.py` — rope config.
- `phase1_train.sh` — canonical launcher (`--target {31b|v6c}`, `--tokens`, `--pack-len`,
  `--pack-512-frac`, `--lr`, `--rank`, `--alpha`).

## 4. Open questions for the council (the audit asks)

1. **Is the 32k-wall RCA right?** Or is the limiter LoRA capacity / module choice / YaRN ramp /
   data? What single experiment most cheaply discriminates? (e.g. eval a 64k-trained stage-2 at
   64k — does Δ drop back to ~0.03, confirming length-wall, or stay high, implicating capacity?)
2. **Is r16/α32 on q/k/o of 5 global layers the right adapter** for long-range *variable
   tracking*? Tracking is a reasoning op; retrieval already holds. Do we need v_proj, MLP,
   higher rank, or more layers — or is attention-range LoRA fundamentally insufficient and a
   different mechanism (full-FT of rope-adjacent params, or longer YaRN training) is required?
3. **Curriculum schedule:** lengths (32→64→128?), token budget per stage, LR per stage, warmup,
   and **continue ckpt-000952 vs restart**. Is staged-length even the right shape vs. one longer
   stage at the max length that fits?
4. **Can 512k be reached by training only to ≤128k + relying on YaRN to extrapolate 128k→512k?**
   The cliff says YaRN extrapolation of a 32k-trained model degrades; would a 128k-trained model
   extrapolate to 512k acceptably, or is genuine ≥256k training mandatory?
5. **bs2 vs cloud decision:** is a 128k-on-bs2 intermediate worth ~a week of GPU time, or skip
   straight to cloud (H100/H200 + ring attention) for a single 256k–512k stage? What's the
   minimum cloud footprint for a 512k LoRA continued-pretrain of a 26B-A4B MoE?
6. **Data adequacy:** current mix (starcoder 0.35 / pg19 0.125 / arxiv 0.125 / slimpajama 0.40,
   min_long 65536). Does this contain enough genuine long-range-dependency signal for VT, or is
   it mostly concatenated-short (which trains position embeddings but not long tracking)?

Audit scope: this brief (§0–3), the RCA (§1), the scaffolding scripts in this directory, and
the curriculum (§3). Point to `path:line` for any claim you challenge.

## 5. AUDIT OUTCOME (council csl-2026-06-08-2145-7680, xhigh; + Claude verification)

Four verdicts. Two were specific checkable claims against artifacts — both verified by Claude
before acceptance (the user is skeptical; claims are not relayed unchecked):

1. **RCA = 32k training-length wall — UPHELD.** All cliff numbers re-verified from live bs2
   `summary.json`: base vt 0.948@32k / 1.0@64k / 0.976@128k / 0.972@256k; ext 0.92 / 0.74 /
   0.648 / 0.62. Apples-to-apples (one template `ruler_native_vt_256k` + `ctx_tokens` retarget).
   - The researcher's "base scores 0.0 at vt_32k" was a **FALSE alarm**: it read a *committed
     placeholder stub* `eval_results_ruler_anchor/ruler_native_vt_32k/gemma-4-26B-A4B-it/summary.json`
     (score 0.0 but served_name/task/num_samples all null — an aborted run, never populated).
     Not live evidence. (Cleanup TODO: that stale stub shouldn't be in the repo.)
   - **Discriminating experiment (accepted):** train a Stage-2 64k smoke (~50M tok) with the
     SAME narrow adapter, eval VT@64k. Δ→~0.03 confirms length-wall AND that the narrow adapter
     suffices *at its trained length* (also partly answers the adapter question). Δ staying ~0.26
     implicates adapter capacity / data signal.

2. **Adapter (r16/α32, q/k/o, 5 layers) — INSUFFICIENT for VT.** ~2M params moves retrieval but
   not variable-tracking (a reasoning op needing MLP state updates; MLP is frozen).
   Recommendation: **widen** to `gate/up/down_proj` + **raise** to r32/α64. (Note: the Stage-2
   discriminator partly tests this — if the narrow adapter recovers VT@64k, "fundamentally
   insufficient" is too strong; the limiter is length coverage, not locus.)

3. **Continue ckpt-000952 via `--resume auto` — BROKEN. VERIFIED.** `total_steps = tokens//
   tokens_per_step` (`phase1_train_yarn_lora.py:646`), cosine built over it (`:650`), resume
   loads OLD optimizer+scheduler (`:669-670`), loop is `range(start_step, total_steps)` (`:755`).
   Same command at step 952/954 → 2 steps at LR≈0. **Worse:** Stage-2 doubles pack_len 32k→64k →
   tokens_per_step doubles → total_steps ~halves to ~477 → `range(952, 477)` is **empty → 0 steps**.
   Trainer has only `--resume {auto,never,must}` (`:434`); **no weights-only-init path exists.**
   → **RESTART each stage:** fresh `--ckpt-dir`, `--resume never`, initialize weights from the
   prior stage's adapter (merge prior stage into base, or add an `--init-from-adapter` flag),
   fresh cosine schedule sized to the new stage's token budget.

4. **512k via ≤128k-train + YaRN extrapolation — NO.** YaRN ramp (β_fast32/β_slow1) leaves
   high-freq dims (49–64, `proportional_yarn_rope_init.py:130-132`) in pure extrapolation; a
   128k-trained model hits a *second* cliff at the 4× stretch to 512k, mirroring today's 32k→64k
   collapse. **Genuine ≥256k training is mandatory → cloud H100/H200 + sequence parallelism.**

### Corrected curriculum
- **Stage 2 (bs2):** 64k packs, MP `--gpus 0,1`, **fresh optimizer/scheduler, weights-init from
  ckpt-000952**, widened adapter (gate/up/down + q/k/o, r32/α64), fresh cosine over the stage
  token budget. Gate: VT@64k Δ≤~0.05.
- **Skip the 128k bs2 stage** — marginal gain, OOM risk, and it does NOT buy 512k (still need
  ≥256k training).
- **Stage 3 (cloud):** ≥256k training with sequence/context parallelism (H100/H200). The only
  path to faithful 512k.
- **Gate before any cloud spend:** run the Stage-2 64k discriminator first — it confirms the RCA
  and the adapter fix cheaply (~8h on an idle bs2 GPU) before committing to cloud.

## 6. REVISED HARDWARE ALLOCATION (council follow-ups #1 csl-…-43ea, #2 csl-…-6618)

**Parallelism verdict:** RING attention is impossible on Gemma 4 (no FA2 online-softmax kernel at
head_dim=512 — upstream FA #2427 open). **ULYSSES** (head-sharding all_to_all on the 5 global
layers' Q/K/V, standard SDPA locally, head_dim-agnostic) is the only context-parallel path.

**Two trainer upgrades unlock 256k on the FREE 2× bs2 box** (move the offload bottleneck off the
critical path), keeping the rental for 512k only:
1. **Async double-buffered offload** — replace the synchronous `save_on_cpu` wrapper
   (`phase1_train_yarn_lora.py:711`) with a CUDA-stream double-buffer that overlaps H2D/D2H copies
   with the next layer's compute. Hides I/O for sliding layers; cannot hide the O(S²) global-attn
   compute (that's the 512k floor).
2. **FP8 (`float8_e4m3fn`) activation-checkpoint compression** before the PCIe hop — 2–4× less of
   the 288 GB/step traffic → I/O 9s→~3s. Does NOT touch the ~26 GB resident SDPA Q/K/V trio peak
   (live compute, not offloaded), so VRAM headroom is unchanged — it only moves throughput.

| Stage | Length | Hardware | Mode | Est. wall | Rental? |
|---|---|---|---|---|---|
| 1 ✅ | 32k | 2× bs2 | DDP | done | no |
| 2 | 64k | 2× bs2 (free) | MP | ~2.5–3 d | no |
| 3 | 128k | 2× bs2 (free) | MP + async-offload | ~3.5 d | no |
| 4 | **256k** | 2× bs2 (free) | MP + async-offload + FP8-compress | ~4.5 d | **no** |
| 5 | **512k** | 4× RTX 6000 (≤1 wk) | **Ulysses** | ~3.2 d | **yes (only this)** |

Notes: PP (pipeline-parallel) is NOT used — our post32k smoke measured PP@64k = 1.00× speedup +
higher VRAM, so MP alone. DiLoCo SKIPPED (data-parallel gradient-sync lever; irrelevant to our
model-parallel activation-memory wall). Wall-times are council estimates — the 64k discriminator
calibrates them. **We do NOT adopt Unsloth** (custom proportional_yarn rope + MoE integration risk);
we extract only its async-offload mechanism into our validated trainer.

### Engineering deliverables (gated on the 64k discriminator confirming the length-wall)
- **E1:** async double-buffered activation offload in `phase1_train_yarn_lora.py` (stream-managed,
  replaces synchronous `save_on_cpu`).
- **E2:** FP8 activation-checkpoint compression hook (cast to `float8_e4m3fn` pre-offload, restore
  on recompute).
- **E3:** Ulysses head-sharding for the 5 global layers (`all_to_all_single` on the head dim) for
  the 4× Stage-5 run.
