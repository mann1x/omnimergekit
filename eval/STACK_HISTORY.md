# STACK_HISTORY — append-only log of `stack.lock.yaml` promotions

Each entry: stack name@version, promoted date, base SHAs of components,
canary verdict on promotion, PRs picked up vs prior version. Cohort
runs reference this log via STACK.txt; HF cards cite version in
footnotes.

See [EVAL_PROTOCOL §v3.3](EVAL_PROTOCOL.md) for the update procedure.

---

## gemma4-moe-stack@2 — 2026-05-21 (proposed, canary pending)

**Status:** built and staged; awaiting v6-coder LCB chain completion
to free the GPU for canary run.

**Components:**
- vLLM: source branch `gemma4-moe-stack-v2`
  - base SHA: `68e07d591` (upstream/main HEAD, 2026-05-21 04:58 EDT)
  - cherry-pick: `3d92852eb` (Fix-E parser hardening, originally `a39e23ed0`)
- lm-eval: 0.4.11 + Fix-A reasoning_content fallback (unchanged from v1)
- modelopt: 0.43.0 (unchanged from v1)

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

**Canary status:** PENDING. Will run `omk_canary.py --stack
stack.lock.yaml --anchor-model google/gemma-4-26B-A4B-it-NVFP4A16
--family gemma-4-26B-A4B` once v6-coder chain releases the GPU.

**Promotion gate:** all 5 structural rules pass AND all 3 anchor
scores (gpqa_diamond, aime24, ifeval) within ±tolerance recorded in
`stack_anchors.yaml`.

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
