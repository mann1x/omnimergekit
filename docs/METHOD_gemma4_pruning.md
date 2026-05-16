# Method: Gemma 4 MoE expert pruning

> **2026-05-16 update**: this doc covers the v3 / v4 baseline (single-class
> aggregation, max-across-classes). For the **v5-targeted** strategy
> (multi-class with class-weights + Tier-B PASS-trace data) see
> [`T17_v5_targeted_pruning_strategy.md`](T17_v5_targeted_pruning_strategy.md).
> For the **T19 aggregation sweep** (13 variants × strategy × weights × protect)
> see [`T19_v5fixed_sweep_results.md`](T19_v5fixed_sweep_results.md). For
> **post-prune router recovery** (the rumination fix) see
> [`T18_moe_router_recovery.md`](T18_moe_router_recovery.md).

Gemma 4 26B-A4B is a 128-expert MoE with top-8 routing per token. The full
F16 model is ~50 GB; Q4_K_M is ~13 GB — too big for 12 GB consumer GPUs.
Pruning experts down to 109 (drop weakest 19 per layer) gets to ~12 GB at
Q4_K_M with **3.5 percentage points** lost on GPQA Diamond (75.25% → 71.72%).

## What's published

- **109e** (`google/gemma-4-26B-A4B-it` → drop 19 per layer): 71.72% GPQA-D.
  Available as `ManniX-ITA/gemma-4-A4B-109e-Q4_K_M-GGUF`. Method: **plain
  REAP** (drop weakest experts by aggregated contribution score, recalibrate
  router on 2-10k tokens).
- **98e** (drop 30 per layer): 67% GPQA-D. More aggressive; trades quality
  for size.

The 109e model is the headline artifact — well-balanced, fits 12 GB Q4_K_M.

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Expert-contribution analysis (gemma4/neuron_analysis/expert_*.py)│
│    On a calibration set, log per-token routing weights × per-expert │
│    activation magnitudes. Aggregate per (layer, expert).            │
│    → expert_neuron_v4.json (~240 MB, per-neuron data)               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. Drop map (gemma4/expert_pruning/generate_drop_map.py)            │
│    Per layer, rank experts by aggregated score; mark bottom-K.      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. Drop & rebuild (gemma4/expert_pruning/expert_drop.py)            │
│    For each layer:                                                  │
│      - delete dropped experts' weight slices                        │
│      - reindex router output dim from 128 → 128-K                   │
│      - reindex shared-expert reference if any                       │
│    Output: a NEW HF checkpoint with smaller config.json             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. Router recalibration (--recalibrate-router 2000)                 │
│    Forward 2-10k calibration tokens through pruned model, record    │
│    surviving-expert routing logits, fit a small linear correction   │
│    on the router so that output distribution matches base on the    │
│    same tokens. Recovers ~50% of PPL loss.                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. Quantize with CD-maps (quantization/quantize_gguf.py)            │
│    CD = Contribution-Dynamic. Per-expert quant tier varies by       │
│    aggregated score from step 1: hot experts get Q6_K, cool         │
│    experts get Q4_K_M. Combined with imatrix → CD-Q4_K_M.           │
└─────────────────────────────────────────────────────────────────────┘
```

## Variants we tried (and what worked)

| Variant | What | Result | Status |
|---------|------|--------|--------|
| **109e drop** | Drop bottom 19/layer, recalibrate router | 71.72% GPQA-D | **published** |
| **98e drop** | Drop bottom 30/layer, recalibrate router | ~67% GPQA-D | published |
| 109e + residual expert | Pack dropped neurons into one extra "residual" expert per layer | Loops on chemistry questions; even with low router scale 0.1, fails REAP Theorem 1 (can't independently gate the packed neurons). | abandoned |
| 109e + DERN (k-means) | Redistribute dropped experts' neurons via spherical k-means + norm equalization | Our k-means averaging destroys per-neuron specialization. Need DERN paper's full norm equalization (Eq. 11) to fix. | abandoned (incomplete impl) |
| 109e Wanda neuron-prune | Structured pruning per neuron | 5-10× worse quality/compression than unstructured. Wrong tool for MoE. | abandoned |
| 109e SVD/MoE-I² rank-reduce | Joint SVD of expert FFN matrices, low-rank approximation | Doesn't reduce bf16 size, degrades quality. | abandoned |
| 124e / 120e hybrid | Drop fewer experts, pack into "hybrid" expert holding multi-source averaged neurons | Marginal gains, complex to maintain. | parked |

## Hard rules learned

1. **128-token analysis budget gives the working 109e**, NOT 1024.
   1024-token captures "stuck reasoning" patterns and ranks experts
   that are bad-at-reasoning higher than they should be. 128-token
   captures "comprehension" — the routing patterns we actually want
   to preserve. The OLD 128-token `expert_neuron_v4.json` is the
   reference, not v5.
2. **Recalibrate the router on 2-10k tokens** after any pruning.
   Without it, expect 5-10 pp PPL increase. With it, ~50% of the
   loss is recovered.
3. **`intermediate_size % 32 == 0`** must hold for expert tensors,
   else Q4_K / Q8_0 fall back to F16 silently. For Gemma 4 this is
   already 1280 (ok). If you frankenmerge onto a different MoE base,
   verify before quantizing.
4. **Always serve Gemma 4 with `--reasoning-format deepseek
   --reasoning-budget 8192`**. Without budget, channel tokens are
   malformed and lm_eval crashes mid-eval at the first chemistry
   question.
5. **`use_cache=True` and `use_cache=False` are TWO branches in
   modeling.py.** Any logic that should affect inference must be
   patched in both. HF `GenerationMixin.generate()` always passes
   `use_cache=True`.

## Recommended order for new compression attempts

Per current research (REAP, Sub-MoE I²-SVD, DERN papers, 2024-2026):

1. **REAP baseline** — pure expert drop + router recalibration. Simplest,
   often already enough. This is what 109e is.
2. **Sub-MoE joint SVD** — stack dropped experts, joint SVD, keep top-r
   singular directions, frequency-weighted V merge. Cleanest math for
   "use the dropped experts somehow".
3. **DERN-style redistribution** — split dropped expert neurons →
   reassign to surviving experts based on cosine similarity → spherical
   k-means with norm equalization within each retained expert. Most
   complex; worth attempting only if REAP+SVD aren't enough.

## Numbers reference

GPQA Diamond, full 198 questions, lm-eval `gpqa_diamond_cot_zeroshot`:

| Model | Score |
|-------|-------|
| 128e original (`gemma-4-26B-A4B-it`) | **75.25%** |
| 109e (drop 19/layer + router recal) | **71.72%** ← shipped |
| 98e (drop 30/layer + router recal) | ~67% |
| E4B (Gemma 4 small, reference) | 57.07% |
