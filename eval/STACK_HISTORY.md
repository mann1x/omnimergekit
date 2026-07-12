# STACK_HISTORY — append-only log of `stack.lock.yaml` promotions

Each entry: stack name@version, promoted date, base SHAs of components,
canary verdict on promotion, PRs picked up vs prior version. Cohort
runs reference this log via STACK.txt; HF cards cite version in
footnotes.

See [EVAL_PROTOCOL §v3.3](EVAL_PROTOCOL.md) for the update procedure.

---

## gemma4-moe-stack@3 — 2026-06-18 (PROMOTED)

**Status:** PROMOTED 2026-06-18 21:00 CEST. Adds a LOCKED `llama_cpp@b9700`
component; no vLLM / lm-eval / modelopt change from @2 (those carry over
unchanged).

**What changed vs @2:**
- llama.cpp promoted from an unpinned procedural `--version` check to a **locked
  stack component at b9700** (`9724f664e803e70eb8d046a3fac411122ad42ff7`), built
  per-arch: sm86 local `/opt/llama.cpp/build` (→ `/shared/dev/llama.cpp-b9700`),
  sm120 bs2 `/mnt/sdc/ml/llama.cpp-b9700`. A llama.cpp bump flips ~1–3% of greedy
  temp=0 problems via CUDA FP-rounding order, so the binary is now versioned like
  vLLM. Supersedes the prior drift (local was b9660).
- New per-model sampling-profile layer (`eval/models/gemma-4.yaml` +
  `eval/sampler_profiles.py`, layered by `omk_eval.py`): greedy stays the
  cross-cohort anchor and the no-flag default is a strict no-op; sampled cohorts
  are selected only via `--sampler` / `--sampler-profile` and tagged in
  `summary.json.sampler`. `bench_policy` default greedy; only
  `gpqa_diamond_full → recommended` (vendor_base).

**Adoption gate (both PASS on b9700):**
- **Anti-loop:** soft2 imat-Q6 @ vendor_minp_rep = **0/48** looping seeds
  (`agentic_loop/results/minp48_soft2_b9700.json`). The published anti-loop
  story does not regress.
- **Anchor:** 128e Q6_K greedy 9-bench within stack-bump tolerance — 7/9 within
  ±3pp of the @2-era `128e_bart` anchor; GPQA −3.54 / AIME −10.0 are small-N
  variance (GPQA-198 ≈7q, AIME-30 = 1 item). The LCB −9.1pp on the plain
  template was a per-slot-ctx truncation artifact (`--parallel 8 × -c 32768` =
  4096 tok/slot starving the 16384-tok request → 9/55 length-capped), **not**
  binary drift: the budgeted `lcb_medium_55_v4` cell came back at **96.36%
  (53/55), above the reference**.

**128e Q6_K greedy vs vendor_base on b9700 (sets `bench_policy`; canonical
`summary.json.score`):**

| bench | greedy | vendor_base | Δ v−g | policy |
|---|---:|---:|---:|---|
| GPQA-198 | 69.19 | 72.73 | +3.54 | **recommended** |
| GSM8K-100 | 94.00 | 86.00 | −8.00 | greedy |
| MATH500-100 | 94.00 | 94.00 | 0 | greedy |
| AIME-30 | 73.33 | 73.33 | 0 | greedy |
| ARC-Challenge | 96.59 | 96.76 | +0.17 | greedy |
| IFEval-100 | 97.00 | 96.00 | −1.00 | greedy |
| HumanEval | 98.17 | 97.56 | −0.61 | greedy |
| HumanEval+ | 91.46 | 92.68 | +1.22 | greedy |
| LCB-55 (v4) | 96.36 | 94.55 | −1.81 | greedy |

Result dirs: `partc_b9700/eval_results_llama_suite/128e_b9700_{greedy,vendorbase}/`.

---

## gemma4-moe-stack@2 — 2026-05-21 (PROMOTED)

**Status:** PROMOTED 2026-05-21 18:07 CEST. All structural rules PASS,
all 3 anchor scores within recorded expectations.

**Components:**
- vLLM: source branch `gemma4-moe-stack-v2`
  - installed version: `0.21.1rc1.dev178+g3d92852eb.cu132`
  - base SHA: `68e07d591` (upstream/main HEAD, 2026-05-21 04:58 EDT)
  - cherry-pick: `3d92852eb` (Fix-E parser hardening, originally `a39e23ed0`)
- lm-eval: 0.4.11 + Fix-A reasoning_content fallback, **refined**:
  content-first; reasoning_content fallback only when content="".
  See `stack.lock.yaml` lm_eval.patches.fix_a_reasoning_content_fallback
  for full rationale.
- modelopt: 0.43.0 (unchanged from v1)
- Vehicle: solidpc RTX 3090 + `--gpu-memory-utilization 0.92
  --max-model-len 32768 --max-num-seqs 4` (graph capture set
  [1,2,4,8]; KV cache 1.65 GiB → fits 32k context)

**PRs picked up vs gemma4-moe-stack@1 (stock 0.20.2):**
- **#42250** — Gemma4 MoE routing closure captures per_expert_scale.
  **This is the rumination root-cause fix.** Was merged to main *after*
  0.20.2 release tag; stock 0.20.2 shipped without it, which is why
  v6-coder hit the IFEval cliff (p50 = 23 528 chars) on stock.
- #43223 — FlashInfer TRTLLM NvFP4 monolithic MoE routing fix
  (`Renormalize → RenormalizeNaive`). Touches NVFP4A16 + softmax+
  renormalize routing.
- #38939 — Add routed experts to openai entrypoint (opt-in API, no
  default-behavior impact).
- #42664 — Normalize reasoning_content → reasoning on requests (input-
  direction only; doesn't change response shape).

**Removed from v1:**
- Stock-0.20.2-only pin removed. The release wheel is now superseded by
  the source-built wheel.

**Canary results (anchor30 on Gemma-4-26B-A4B-it-NVFP4A16, 30 questions,
greedy + thinking_token_budget=12288):**

| sub-bench | score | structural | anchor |
|---|---:|---|---|
| anchor_gpqa_10 | 9/10 | PASS | recorded ±0.20 |
| anchor_aime_10 | 7/10 | PASS | recorded ±0.20 |
| anchor_ifeval_10 | 10/10 | PASS | recorded ±0.20 |
| **VERDICT** | — | **ALL_PASS** | — |

Run dir: `eval_results/canary/stack_v2_20260521_171905_fix-a-refined/`.

**Bug discovered + fixed during canary:** original Fix-A behavior
concatenated `reasoning + "\n" + content` whenever both were populated.
With thinking-on this contaminated IFEval rule-based scorers (no-comma,
lowercase, multi-section) — first canary run scored IFEval **0.10** on
the same model that scored **1.00** after the refined Fix-A patch. The
content-only behavior preserves silent-empty rescue while keeping
short-answer scorers honest.

**Wheel build status (2026-05-21 18:07):**
- sm86 (RTX 3090, A40) — ✓ built + installed in `vllm` env
- sm89 (RTX 4090, L40, RTX 6000 Ada) — ✓ built
- sm90 (H100, H200) — building (flashmla cutlass submodule init was needed)
- sm100 (B100/B200) — queued
- sm120 (RTX 50-series) — queued

---

## gemma4-moe-stack@1 — 2026-05-14 (RETIRED)

**Status:** retired 2026-05-21 — failed structural canary post-hoc.

**Failure mode:** stock vLLM 0.20.2 release wheel does not include
`#42250` (Gemma4 MoE closure-capture fix). On pruned-MoE Gemma 4
variants, per-expert scale captures stale in the routing closure →
routing distribution drifts → model ruminates → response p50 explodes
to 10 000–23 000 chars on benches that should answer in 600–700.

**Empirical record (structural canary against retired stack):**

| sample (model × bench) | n | p10 | p50 | p99 | verdict |
|---|---:|---:|---:|---:|---|
| 128e × IFEval-100 (good) | 100 | 120 | 855 | 13060 | OK |
| v4 × IFEval-100 (good) | 100 | 127 | 837 | 18396 | OK |
| **v6 × IFEval-100 (broken)** | 100 | **4106** | **23528** | **87039** | **FAIL** |
| v5-it × HE-164 (broken) | 164 | 2752 | 10492 | 43247 | FAIL |
| v6 × HE-164 (broken) | 164 | 2756 | 10526 | 53186 | FAIL |
| v6 × HE+-164 (broken) | 164 | 2669 | 9593 | 67431 | FAIL |

128e and v4 *pass* the canary on this stack because their routing is
robust enough to not get pushed into the closure-capture pathology.
Aggressively pruned variants (v5, v5-coder, v6) progressively fail it.
This is the failure mode v2 fixes by including #42250.
