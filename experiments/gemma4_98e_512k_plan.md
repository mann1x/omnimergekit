# Gemma 4 26B-A4B 98e-v5 → 512k context (no-quality-loss target)

**Status:** plan / pre-launch — armed for the day the Linode token lands.
**Owner:** mannix
**Drafted:** 2026-05-21
**Supersedes:** initial 1M brief (research only — 1M target rejected due to expected MRCR-8 quality drop 44.1% → 20-30%; not acceptable under user's constraints)

## Goal & hard constraints

- **Context:** extend `mannix/gemma4-98e-v5` from native 256k → **512k** (`max_position_embeddings = 524288`).
- **No-quality-loss gate (must hold, not a goal):**
  - Canonical 9-bench at greedy / 32k context: each bench within ±1pp of the v5 stack@2 baseline (the same numbers cited on the published HF card). Per-bench tolerances respect existing recorded noise (e.g. AIME-30 ±3pp because n=30; GPQA-198 ±1pp; HE-164 ±1pp).
  - RULER NIAH single-needle @ 32k: within ±2pp of pre-extension number.
- **Long-context goal (not a hard gate, but the experiment is uninteresting without it):**
  - RULER NIAH single-needle @ 256k: ≥90% (anchor: v5 native 256k pre-extension number, measured first).
  - RULER NIAH single-needle @ 512k: ≥85%.
  - MRCR-v2 8-needle @ 256k ≥ 40% (anchor base 26B-A4B-it = 44.1% @ 128k).
- **MoE routing health:** routing entropy at 256k and 512k positions must remain ≥0.85× the entropy measured at 32k for every layer. Any single expert dominating >40% of tokens at long range = abort.

Why 512k not 1M: 1M is a 4× factor on top of 256k native and the closest analog (Llama-3-8B-Gradient-1048k) drops single-needle NIAH from ~95% → ~80% even at native arch (no pruned-MoE risk on top). 512k is a 2× factor — well within YaRN's documented "lossless" envelope (≤2× before noticeable degradation per the YaRN paper §5.2).

## Architectural reminder (do not skip)

Gemma 4 has **hybrid attention**: 25 of 30 layers are sliding-window (`sliding_attention`, window=1024, RoPE θ=10000); 5 are full-attention (`full_attention`, layer indices `[5, 11, 17, 23, 29]`, RoPE θ=1000000, head_dim=512, num_global_kv_heads=2). **The sliding layers do not need extension.** The window doesn't grow regardless of context length, so their RoPE base stays correct at any context. The extension applies *only* to the 5 full-attention layers.

The 98e variant inherits this exactly — pruning touched experts (FFN), not attention.

## Recommended recipe

**YaRN factor = 2.0** on `rope_parameters.full_attention` only (5 layers), with a **conservative continued pre-train LoRA** (250-300M tokens, low LR, aggressive checkpointing, no MoE touching).

### Why YaRN at 2.0 and not higher

- The YaRN paper's "lossless" regime is ≤2×. We're exactly at that boundary.
- Stays clear of LongRoPE-style per-dim search (which adds calibration complexity, infra burden, and a tuned hyperparameter we'd have to defend).
- Easier to revert: if the 32k canonical 9-bench regresses post-bake, we can blend the LoRA back at α<1 instead of throwing the run out.

### Why a fine-tune at all

A *no-fine-tune* NTK-aware approach would extend to 512k but degrade short-context (the unmodified θ + rescaled positions misalign at small contexts). The no-quality-loss gate at 32k requires the model to *see* both short and long examples during training. PG-19's natural distribution gives both.

### LoRA scope

- **Adapt:** Q, K, V, O projections of the **5 full-attention layers only** (`layer_types[5,11,17,23,29].self_attn.{q,k,v,o}_proj`).
- **Do NOT touch:** FFN/MoE experts (would unwind the entire v5 pruning + α=1.2 shared-upweight work), embeddings, lm_head, sliding-layer attention.
- LoRA rank r=16, α=32, dropout=0.0.
- Trainable params: ~14M (LoRA on 5 layers × 4 projections × bf16 base).

## Compute & hardware

| Item | Value |
|---|---|
| Training GPU | **1× H100 80GB** (preferred; 1× H200 141GB equally fine). L40S 48GB possible with ring-attention but adds infra burden — skip. |
| Training duration | ~6-10 h for 250M tokens at packed seqlen 256k on H100 (~12k tok/s effective with FlashAttention-2 + paged optimizer states) |
| Training cost (Linode) | ~$25-40 |
| Training cost (Vast spot, fallback) | ~$10-18 |
| Inference (eval) GPU | 1× H100 80GB single-stream; KV at 512k for 5 global layers = 10.2 GB bf16 / 5.1 GB fp8 / 2.6 GB NVFP4. With NVFP4A16 weights (14 GB) + fp8 KV, fits comfortably. |
| Disk floor | 200 GB (HF cache 60 GB + checkpoints 5×50 GB peak before pruning the loser checkpoints + working space for eval logs) |

## Calibration / training data

- **Primary:** `emozilla/pg19` (PG-19, 2B clean book tokens) — repack into 256k-512k packed sequences (~10-15k books per mega-doc).
- **+10% math tail:** `EleutherAI/proof-pile-2`, the `algebraic-stack` + `arxiv` subsets, packed identically.
- **+10% prose-distribution match:** `togethercomputer/RedPajama-V2` long-doc subset (filter for doc length ≥64k).
- **Target seen tokens:** 250-300M (vs the original 1M plan's 100-200M — buying more headroom for the no-quality-loss constraint).
- All three packed into 256k chunks with 50% chance of further pack-up to 512k per chunk (curriculum: model sees both lengths interleaved).

## Two-phase pre-flight (BEFORE fine-tune launches)

**Phase 0 must complete cleanly before phase 1 even compiles.**

### Phase 0 — anchor measurements on solidPC (no Linode burn)

1. Run `omk_eval` 9-bench canonical greedy on v5 NVFP4A16, **at 32k context** (current default). Record per-bench score with full tolerance. This is the regression bar.
2. Run RULER NIAH single-needle on v5 at 32k, 64k, 128k, **256k** (native; will need `--max-model-len 262144` + corresponding KV memory). Pick a 256k-capable serving config — the published vLLM stack@2 supports it.
3. Run MRCR-v2 8-needle @ 128k. Anchor for the long-context goal.
4. Verify v5 actually does 256k: launch vLLM with `--max-model-len 262144`, fire one long-context prompt (the LongBench-v2 longest), inspect output for coherent answer (not just non-empty). If it ruminates or fails to track entities, native 256k is *not* really there and we have a deeper problem before any extension.

If phase 0 anchors land cleanly, proceed. If not, root-cause first.

### Phase 1 — vLLM stack@3 patch (independent of Linode)

The current stack@2 vLLM does not handle YaRN-aware scaling on Gemma 4's nested `rope_parameters` schema. Required patch:

- Patch `vllm/model_executor/models/gemma4.py` to read `rope_parameters.full_attention.yarn_scaling_factor` and `original_max_position_embeddings`, then apply YaRN scaling to **layers in `layer_types == "full_attention"` only**.
- Land it as a third cherry-pick on top of stack@2 (cherry #3 = YaRN-hybrid). New version = `gemma4-moe-stack@3`.
- Test the patch on the **unextended** v5 first — applying YaRN with factor=1.0 should be a no-op. If the 32k canonical 9-bench drifts at factor=1.0, the patch has a bug.
- Land it through the existing canary process: structural canary (anchor30) + the reference-anchor expected scores from `stack_anchors.yaml`.

## Step-by-step experiment plan

When the Linode token arrives:

1. **Provision Linode H100** (or fall back to Vast H100 spot if Linode inventory is dry — typical Akamai bottleneck). Bootstrap via the canonical `pod_bootstrap_reeval.sh` + `pod_setup_eval_envs.sh` + add `gemma4-moe-stack@3` wheel (built on solidPC during phase 1).
2. **Smoke gate**: re-run anchor30 (3 sub-benches × 10 q) on the pod's v5 NVFP4A16. Must match solidPC anchor30 within ±2 questions. If not, the stack didn't bootstrap cleanly — fix before training.
3. **Build packed training shards** on the pod (PG-19 + ProofPile-2 + RedPajama-long). Save to `$WORKSPACE/train_packed/`. ~30 min for 250M tokens at 256k pack-len.
4. **Launch LoRA fine-tune** (`omnimergekit/recipes/gemma4/lora_extend_512k.py` — TO BE WRITTEN; sketch in §"Training script" below). Save checkpoint every **25M tokens** (10-13 checkpoints total).
5. **Per-checkpoint guards (in-loop, automated):**
   - Routing-entropy probe: 1k tokens at positions [32k, 128k, 256k, 512k], per-layer top-8 gate distribution. Abort if any layer's top expert >40% at any of those positions OR if entropy at 256k <0.85× of 32k entropy.
   - 32k canonical 9-bench mini (5 q × 9 benches = anchor30-style structural canary). Abort if score drops >2pp from phase 0 baseline.
6. **Checkpoint selection**: among checkpoints that survive in-loop guards, run full 32k canonical 9-bench. Pick the *first* (oldest) checkpoint that matches v5 baseline within ±1pp. We prefer less training — minimum perturbation, minimum drift risk.
7. **Bake** the selected LoRA into bf16 weights. **CRITICAL:** `save_pretrained(max_shard_size="10GB")` (cgroup-OOM lesson from he1v2; default 50GB will kill it on 62GB-RAM pods).
8. **Patch `config.json`**: `rope_parameters.full_attention.yarn_scaling_factor = 2.0`, `rope_parameters.full_attention.original_max_position_embeddings = 262144`, `max_position_embeddings = 524288`. Keep `rope_parameters.sliding_attention` UNCHANGED.
9. **Long-context eval at 512k**:
   - RULER NIAH single-needle @ {32k, 64k, 128k, 256k, 512k}.
   - MRCR-v2 8-needle @ {128k, 256k}. 512k MRCR is too noisy at this scale (the 8 needles + 512k context exceeds practical NIAH design; skip unless explicitly requested).
   - LongBench-v2 (if we have time/budget; secondary).
10. **Short-context regression eval**: full canonical 9-bench at 32k. **Hard gate:** every bench within ±1pp of v5 baseline. If any single bench fails the gate, the run is rejected and the recipe goes back to the drawing board.
11. **Quantize**: `quantize_any.py --method nvfp4a16` on the baked bf16. Re-eval anchor30 at 32k + RULER 256k on the NVFP4A16 to confirm quant didn't break long-context (separate quant-induced risk).
12. **Publish to HF + ollama**: as `mannix/gemma4-98e-v5-512k` (new repo, do NOT overwrite v5). Card cites stack@3, training mix, all the above eval tables.

## Training script sketch (TO BE WRITTEN at phase 1)

`omnimergekit/recipes/gemma4/lora_extend_512k.py`. Skeleton:

```python
# inputs:  /workspace/v5_bf16/                (HF model dir, native 256k)
#          /workspace/train_packed/           (PG19+ProofPile2+RedPajama long, 256k packs)
# outputs: /workspace/checkpoints/ckpt_{N}/   (every 25M tokens)
#
# - load with attn_implementation="flash_attention_2"
# - apply LoRA r=16, α=32 on Q/K/V/O of layers [5,11,17,23,29] ONLY
# - patch model.config.rope_parameters.full_attention before forward pass
#   to apply YaRN factor=2.0 (so training sees the post-extension RoPE)
# - cosine LR 2e-5 → 2e-6 over 250M tokens, warmup 5M
# - grad accum to fit 256k packed sequences in 80GB (likely seqlen=1, bsz=1)
# - eval guard hook: every 25M tokens →
#     * routing-entropy probe
#     * 5q × 9-bench mini
#     * write checkpoint + summary.json
#     * abort signal file if any gate fails
# - LoRA r=16 is light enough that even on H100 the bottleneck is paged-KV
#   for the 256k forward pass, not param compute.
```

## Risk register

- **MoE routing drift at long range:** unmapped territory. Best mitigation = the in-loop routing-entropy probe + pick-earliest-passing-checkpoint policy. Worst case: the run consistently fails the entropy gate. Fallback: reduce YaRN factor to 1.5 (extend to 384k only, not 512k); accept smaller win.
- **CD-map calibration limit:** the contribution map that chose which 30 experts to drop was scored at ≤8k contexts. A previously-dropped expert that's load-bearing at 256k+ shows up as: a surviving expert dominates routing for long-range token positions, OR perplexity explodes at long range while short-context perplexity stays clean. The probe catches it.
- **vLLM stack@3 patch bugs:** the patch is small but un-tested in the wild for Gemma 4's nested schema. Mitigation: phase 1 patch validation on unextended v5 with factor=1.0 catches schema mismatches before any expensive training.
- **Quantization-induced long-context regression:** NVFP4A16 has known minor effects at long context. Mitigation: step 11 re-evals long-context post-quant; if it drops >5pp on RULER @ 256k vs bf16 we publish bf16-only and note NVFP4A16 limitation.
- **Linode H100 inventory:** Akamai Cloud has historically been RTX-6000-dominant. Confirm H100/H200 in the user's region *before* launching the training step. Have Vast H100 spot as fallback in the bootstrap script.

## Open decisions / things to lock before launch

1. **Hard floor on the 32k regression gate: ±1pp per bench, or tighter?** Currently ±1pp; tighter would slow checkpoint selection. ±1pp is published-noise-tolerant.
2. **Final eval suite at 512k:** RULER + MRCR + LongBench-v2, or RULER + MRCR only? LongBench-v2 is ~5h extra; skip unless we have publishing-table budget.
3. **Hub destination:** `mannix/gemma4-98e-v5-512k` (new repo) confirmed? Naming convention — do we want `-512k` or `-long` or `-extended-512k`?
4. **Do we publish a GGUF for this?** Probably no for v1 (llama.cpp's Gemma 4 path is broken per project memory). Note in card: "vLLM-only at launch."

## Files this plan will produce

- `omnimergekit/recipes/gemma4/lora_extend_512k.py` — training script (phase 1 work)
- `omnimergekit/eval/templates/ruler_niah.yaml` — RULER NIAH eval template (likely exists in lm-eval; add a long-context-stride wrapper)
- `vllm/model_executor/models/gemma4.py` patch — YaRN-on-full-attention only, stack@3 cherry-pick
- `stack.lock.yaml` bump to v3 + STACK_HISTORY.md entry
- HF repo `mannix/gemma4-98e-v5-512k` with card citing all eval tables

## Memory notes to add on completion (good outcome)

- `project_98e_512k_extension.md` — recipe + scores + the routing-entropy probe behavior at long range (this is publishable signal regardless of outcome).
- Update `feedback_pod_image_canonical.md` if stack@3 changes the pod bootstrap recipe.

## References

- YaRN: https://arxiv.org/pdf/2309.00071
- LongLoRA: https://arxiv.org/html/2309.12307v3
- Gemma 4 architecture (config.json): https://huggingface.co/google/gemma-4-26B-A4B
- Llama-3-8B-Gradient-1048k receipt: https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k
- Qwen2.5-14B-Instruct YaRN config (reference for YaRN-in-config schema): https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
- vLLM long-context docs: https://docs.vllm.ai/en/latest/features/context_extension/
- vLLM hybrid KV cache: https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/
- RULER: https://github.com/NVIDIA/RULER
- MRCR-v2 (Google benchmark): https://github.com/google-deepmind/mrcr
