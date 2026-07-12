# Qwen3.6-27B-Omnimerge-v4 — Sources & Evaluation Provenance

Companion doc to [`pod_omnimerge_v4_build.sh`](pod_omnimerge_v4_build.sh) and
[`pod_v4_q6k_eval_chain.sh`](../../scripts/pod_v4_q6k_eval_chain.sh). Records **which models
were merged**, at **what weights**, and **exactly what was and was not measured** — including
the eval hardware. Written to close the "source evaluation" documentation gap.

> ⚠️ **Name collision warning.** `Qwen3.6-27B-Omnimerge-v4` (this merge) is **unrelated** to
> `gemma-4-A4B-98e-v4` (a Gemma-4 26B-A4B MoE expert-drop surgery variant). The local
> `backup_models/eval_results_v4/` tree on solidpc holds the **Gemma-4 98e-v4** results, *not*
> this merge. Do not cross-read them.

## Merge

- **Base:** `Qwen/Qwen3.6-27B`
- **Method:** `omnimerge_v2` — DARE-TIES base + OBIM-lite + DAREx q + EMR election
- **Params:** density `0.53`, DAREx q `0.75`, seed `42`
- **Post-merge surgery:** MLP-passthrough — `mlp.{gate,up,down}_proj` copied verbatim from clean
  Qwen3.6 base (defends Qwen3.6's fragile reasoning-tag policy; see the model card's
  "Key finding" section). Everything else (attn, linear_attn, norms, embed/head) is from the merge.

## Sources

| Source | Weight | Role | Individually benchmarked? |
|---|---|---|---|
| [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) | base | base + chat template | yes — appears as the "Qwen3.6 base" column below |
| [`rico03/Qwen3.6-27B-rico03`](https://huggingface.co/rico03) (Claude-Opus reasoning distill) | 0.40 | general capability | **no** |
| [`ValiantLabs/Qwen3.6-27B-Esper3.1`](https://huggingface.co/ValiantLabs) | 0.35 | code + reasoning | **no** |
| [`kai-os/Qwen3.6-27b-Opus4.6-reasoning`](https://huggingface.co/kai-os) (LoRA → base anchor) | 0.25 | reasoning anchor | **no** |

### ⚠️ The three source fine-tunes were never benchmarked individually

There is **no per-source eval script or result** for the Qwen3.6 (v4) source fine-tunes anywhere in
this repo or on solidpc. Only the **merged** model and the **base** were scored. The source weights
(0.40 / 0.35 / 0.25) were assigned from the sources' *published/role* profiles + the v2 precedent,
**not** from an in-house per-source benchmark on this generation.

The only per-source eval that exists in the repo is for the **previous generation** — the
**v2 / Qwen3.5** source set (`pod_qwen3_5_eval_sources.sh` → `humaneval_sources.log`), which is a
**different set of models** and HumanEval-only. For reference (Qwen3.5, HE pass@1, not v4):
`Claude-4.6-Opus-Reasoning-Distilled 84.76` · `Esper3.1 83.54` ·
`Gemini3-Pro-High-Reasoning-Compact-Thinking 83.54` · `Writer-V2 82.32`.

To fill this gap, a `pod_qwen3_6_eval_sources.sh` (mirror of the v3.5 sources script, retargeted to
the three Qwen3.6 fine-tunes, HE + MBPP + GPQA at the canonical greedy Q6_K recipe) would need to be
written and run. **Not yet done as of this doc.**

## Merged-model scores (Q6_K, canonical greedy)

Greedy (`do_sample=False, T=0.0, top_p=1.0, top_k=0`), raw `/v1/completions` on llama.cpp with
`--reasoning-format deepseek --reasoning-budget 8192 --parallel 2 -c 65536` + q8 KV. HE/MBPP
think-stripped before `exec`.

| Bench | Qwen3.6 base Q6_K | **Omnimerge-v4** | Δ vs base | Δ vs Omnimerge-v2 (Qwen3.5) |
|---|---|---|---|---|
| HumanEval (164q) | 84.76% (139/164) | **83.54%** (137/164) | −1.22 pp | +4.27 pp |
| MBPP (500q, corrected) | 57.60% | **73.00%** (365/500) | +15.40 pp | −1.60 pp |
| GPQA Diamond (198q, flexible-extract) | not measured | **78.28%** (155/198) | — | +9.09 pp |

MTP companion (`-MTP-GGUF`) is statistically indistinguishable (HE 137/164 ↔ 137/164,
GPQA 155/198 ↔ 154/198) at 2.0–2.3× decode speed.

## Eval hardware — **cloud pod, not solidpc**

The canonical Q6_K eval chain (HE → MBPP → GPQA, standard **and** MTP) ran on a **rented cloud pod
(vast/runpod id `37268930`)**, then rsynced results back to solidpc — see the header of
[`pod_v4_q6k_eval_chain.sh`](../../scripts/pod_v4_q6k_eval_chain.sh) (`# - rsyncs results back to
solidpc`). GPQA wall time was **4 h 55 min on the pod's 3090-class GPU**. The "3090" in the model
card is the **pod's** GPU, not solidpc's — these headline numbers are **not** a solidpc-native run.
(Only the archived result files live on solidpc.)

The 2026-05-22 canonical greedy re-run patched lm-eval's `api_models.py:545` `UnboundLocalError`
(transient `TimeoutError` before `outputs` assigned) + an aiohttp lifecycle bug — see
[`pod_v4_q6k_gpqa_chain.sh`](../../scripts/pod_v4_q6k_gpqa_chain.sh). The earlier `≈ 84.75%` GPQA
figure (partial 177/198, sampled T=0.6, budget 16384) is **superseded** by the 78.28% full greedy.
