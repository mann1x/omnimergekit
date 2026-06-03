# Gemma-4 26B-A4B — 62e compression & capability redistribution (PARKED)

**Status:** PARKED 2026-06-03. The 28e/62e aggressive-prune line and the
capability-redistribution framework built on top of it are stopped. This file
is the accurate, complete log of every experiment, mechanism, recipe, result,
and verdict so the negative is reproducible and the tooling is reusable.

**One-line conclusion:** at a fixed 62-expert budget, *no* redistribution lever
— closed-form fold or trainable KD — recovers a capability that pruning
destroyed. Capability recovery at a fixed budget is a **selection** problem
(solved by the v5/v6-coder keep-set search), **not** a redistribution problem.

---

## 0. Subject

- **Base:** `google/gemma-4-26B-A4B-it` — MoE, 128 experts/layer, top-8 routing,
  30 layers, `moe_intermediate_size=704`, `hidden=2816`, bf16.
- **A2** = `gemma-4-A4B-62e-fc15_25-p8-pes120-it` — 128e→62e aggressive prune
  (floor-clamp 15–25, protect-top 8, per-expert-scale α=1.20), fits ~12 GB VRAM.
- **The wall:** A2 has a residual degenerate-loop floor on a 200-prompt greedy
  bf16 screen (`loop_screen.py`, max-new 2048):
  **15.5 % (31/200)** — constrained 6/80, **multilingual 21/60**, openended 2/56,
  seeds 2/4. A2 stayed UNSHIPPED until this floor was resolved.

Hardware: **blackswan-2** (Linode, 2× RTX PRO 6000 Blackwell 97 GB). All builds
and evals below ran there unless noted. Eval sampler **frozen greedy** throughout
(temp 0 / top_p 1 / top_k 0 / do_sample false), the cohort-wide apples-to-apples rule.

---

## 1. Selection is exhausted (why redistribution was the only lever left)

Before redistribution we proved no *selection* of 62 experts moves the loop floor.

| probe | what it tested | result |
|---|---|---|
| **T175** loop-floor sweep (66e/68e/72e) | does raising the budget a little help? | floor flat across 62→72e — count is the wrong lever |
| **T187** force-keep 8 format experts | constrained-format is localized? | YES — bucket halved 11→5 (a few experts carry it) |
| **T188** methodology audit | why did every anti-loop lever fail? | loops are not a selection-reachable defect at 62e |
| **T189** multilingual router differential + clean force-keep | is multilingual localized? | NO — lost routing mass spread over **1956/1980** (layer,expert) cells; force-keeping the top-16 moved the bucket 27/60→26/60 (NULL) |

**Mechanistic split (the load-bearing finding):** *localized* capabilities
(constrained-format, code) live in a few experts and are stable/recoverable;
*diffuse* multilingual is spread over essentially the whole expert set and
exceeds the low-rank subspace 62 survivors can represent. The 128e teacher
answers the same prompts fluently, so the loss is real prune-induced capability
loss, not a measurement artifact.

Conclusion: selection is exhausted; the only remaining lever is **redistributing
the excluded experts' function INTO the survivors.**

---

## 2. The redistribution framework (`redist.py`)

A driveable 5-stage pipeline — `localize → capture → redistribute → gate → build`
— where **driver = (capability spec + corpus)**. Swap the driver corpus → a
different recovered 62e. Two redistribution arms:

- **Arm A — closed-form fold** into A2's FIXED 62 survivors + fixed router:
  HC-SMoE (output-space freq-weighted cluster-merge), MergeMoE (per-survivor
  least-squares `T = Q·P†`), REAM (Router-weighted Expert Activation Merging;
  saliency = gate-mass × ‖mean-out‖).
- **Arm B — trainable survivors** via teacher-128e logit forward-KL KD
  (extends `router_kd.py`'s `select_router_params` to expert/router tensors).

Gating rule (non-negotiable, the off-manifold rule): **gate on `loop_screen`
AR generation and on the held-out code benches, NEVER on KD/reconstruction loss.**
Reconstruction loss does not predict AR quality (verified repeatedly:
`feedback_optimizer_off_manifold`, `feedback_calibration_signal_misleading`).

---

## 3. Arm A + B results — diffuse multilingual (T191/T192)

Every lever converged to **≥ A2's loop floor**, never below. All cells
`loop_screen` 200-prompt, greedy bf16, max-new 2048. (overall % · multilingual /60)

| mechanism | cell | loop % | ML /60 |
|---|---|---|---|
| anchor | **A2** | **15.5** | **21** |
| closed-form | REAM | 18.0 | 23 |
| closed-form | HC-SMoE | 19.0 | 24 |
| closed-form | MergeMoE | 21.5 | 28 |
| expert-KD (mid L8-22) | lr 1e-4 | 27.5 | 35 (catastrophic) |
| expert-KD (mid) | lr 2e-5 | 18.5 | 24 |
| expert+router-KD (mid) | balanced corpus | 19.0 | **20** (best ML anywhere, ≈ A2's 21) |
| expert+router-KD (mid) | 400 steps | 21.5 | 27 |
| **shared dense branch** (`mlp.*`, routing-independent) | r3_shared | 23.0 | 29 |
| **outer layers** (L0-7 + L23-29) | r3_outer | 21.0 | 30 (held constr=6, oe=2) |
| trainable rank probe | **E-RankProbe** | — | **CAPACITY_WALL ~26 % rel-MSE plateau** (flat 250→1500 steps) |

Reads:
- **lr is the dominant off-manifold knob** (1e-4 catastrophic).
- More steps drift ML *worse* (400-step cell), not better.
- The trainable router **reconstructs** A2, never exceeds it.
- **r3_outer** preserved the localized buckets (constr/oe at A2 levels) and only
  worsened diffuse multilingual — the cleanest confirmation of diffuse-vs-localized.
- **E-RankProbe** is the definitive diagnostic: 62×704 survivors cannot span the
  128-expert blended teacher output for a diffuse signal, regardless of objective
  (the plateau holds even with a frozen router → the bottleneck is expert weight
  span, not gating).

**Council adversarial sign-off** (`csl-2026-06-02-2050-c4f7`, xhigh, researcher
grounded in source): the negative is STRUCTURAL and PUBLISHABLE. Untested levers
ruled out a priori — true all-30-layer KD (vocab-chunking solves OOM not capacity);
learned low-rank router delta (a routing-selection lever, not a capacity lever);
shared dense branch (~1.6 % of survivor capacity/layer, structurally insufficient).
Recommendation: declare diffuse-multilingual OUT-OF-SCOPE at 62e; pivot `redist.py`
to LOCALIZED capabilities where closed-form provably recovers.

---

## 4. Arm A + B results — localized code (T193 / T193b)

Per the council rec we pivoted the driver to the *localized* code capability,
driven by 360 code prompts+completions (HumanEval / HumanEval+ / LCB pass-traces).
Apples-to-apples anchors = A2 (pes120) Q6_K-llama: **HE+164 0.8963 / MPE-100 0.7533**.

| cell | bf16 loop % (constr/ML/oe) | HE+164 | MPE-100 | verdict |
|---|---|---|---|---|
| **A2** anchor | 15.5 (6/21/2) | 0.8963 | 0.7533 | — |
| **T193 REAM fold** | **19.0** (8/23/5) | 0.8841 (−1.2) | 0.7200 (−3.3) | REGRESSED |
| **T193b code-KD** (experts+router, mid, 200 st, lr 2e-5) | **20.0** (11/24/2) | 0.9024 (+0.61) | 0.7600 (+0.67) | ship-gate FAIL |

Reads:
- **REAM fold regressed code, unconfounded.** The bf16 loop_screen (no quant/imatrix)
  rose to 19.0 % and the code benches dropped. A2's `fc15_25-p8` keep-set was
  *selected to preserve code* → the 66 dropped experts are the least code-relevant
  → there is no lost code to recover; folding the residue is pure perturbation, and
  a fixed router cannot independently gate the merged function (**REAP Theorem 1**).
- **code-KD is the only mechanism that moved code in the right direction**
  (HE+ +0.61, MPE +0.67) — so 62 survivors *do* hold spare capacity for a
  localized capability. But (a) the gains are 1-problem deltas (1/164, ~1/100),
  at/below noise; and (b) they cost a real cross-bucket regression —
  constrained-format loops **doubled, 6→11/80**, overall loop 15.5→20.0 %.
  Training shared mid-layer (L8-22) experts cannibalizes the capacity that
  constrained-format relies on. The KD loss fell to ~0.8 (off-manifold corpus
  memorization), exactly why the gate is AR-screen, not loss.
- Ship-gate (loop materially < A2 **and** HE+ ≥ 0.899 / MPE ≥ 0.757 **and** no
  cross-bucket regression): code-KD clears the code bars but fails loop +
  no-regression. Net trade = +1 code problem for +9 loops.

---

## 5. Verdict

The redistribution-into-a-fixed-(router,survivor)-set line is **fully exhausted**
across all three families:

1. **Selection** — exhausted (T187-189).
2. **Closed-form fold** — regresses on *both* diffuse-multilingual *and* localized-code.
3. **Trainable KD** — *reconstructs* on multilingual (never exceeds A2),
   *noise-trades* on code (gains one problem, loses coherence).

Unifying mechanism: a fixed router cannot independently gate added function
(REAP Thm 1), and trainable KD on shared experts trades one capability for
another. **Capability recovery at a fixed 62-expert budget is a SELECTION
problem** — the v5/v6-coder line beats A2 on code by *choosing a better keep-set*
(v6-coder C6v3lcb Q6_K: HE+164 0.9329 vs A2 0.8963), not by folding or training
into A2's survivors.

Open directions (user's call, not pursued here): (a) accept A2's floor & ship;
(b) raise the expert budget above 62 where selection has headroom (T175 territory);
(c) conclude & write up the structural negative.

---

## 6. Tooling & scripts (blackswan-2 unless noted)

Framework + drivers (`/srv/ml/scripts`, mirror of `/srv/ml/repos/omnimergekit/scripts`):
- `redist.py` — the 5-stage CLI (localize/capture/redistribute/gate/build); REAM/HC-SMoE/MergeMoE methods.
- `redist_localize_divergence.py` — fluent-failure (teacher-correct-vs-student-failing) localization signal.
- `redist_rank_probe.py` / `redist_rankprobe_run.sh` — E-RankProbe capacity test.
- `redist_run.sh`, `redist_smoke_closedform.sh` — closed-form driver runners.
- `redist_expert_kd_run.sh`, `redist_code_kd.sh` — trainable-KD runners (extend `router_kd.py`).
- `redist_code_fold.sh`, `redist_code_eval.sh` — T193 code-fold build + HE+164/MPE-100 eval.
- `loop_screen.py` — 200-prompt greedy AR loop canary (the gate).

Corpora: `/mnt/sdc/ml/corpora/{loop_screen_sample.jsonl, redist_calib_code.jsonl,
kd_corpus_code.jsonl, kd_corpus_ml_heavy.jsonl}`.

Results (preserved): `/srv/ml/eval_results_redist/` (loop_*.json + code_eval summaries),
REAM capture `/srv/ml/redist_work/capture_code_ream.pt`.

**Note on REAM/Thm 1:** the `redist.py` REAM docstring claiming it "sidesteps REAP
Theorem 1" is conceptually WRONG — REAM changes merge weighting but a fixed router
still cannot independently gate the folded function. Fix that comment before any
upstream commit of the method.
