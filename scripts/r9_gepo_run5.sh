#!/usr/bin/env bash
# R9 run5 -- GEPO-ENTROPY (arXiv 2607.16850). The first run of the method the paper
# actually describes, on the pool run4 already proved safe.
#
# TWO METHODS SHARE THE ACRONYM. READ THIS BEFORE TABLING run5 AGAINST run1-run4.
# ---------------------------------------------------------------------------------
#   arXiv 2508.17850  Group EXPECTATION Policy Optimization -- a GSPO-family
#                     IMPORTANCE WEIGHT. GSPO's per-sample denominator q_i is replaced
#                     by the group expectation E_q[q] = sum(q^2)/sum(q). run1-run4 ran
#                     this, and so did the AN / Gemma-4-E2B control that WORKED
#                     (an-finetune/simpo/gepo_trainer.py:2 says so in its own header,
#                     and train_gepo_v16.sh passes no entropy flags).
#   arXiv 2607.16850  Group ENTROPY-CONTROLLED Policy Optimization (Cheng, Lyu, Gao,
#                     Zhang, Chen; Shanghai AI Lab; 2026-07-21) -- an ADVANTAGE SHAPING
#                     rule bolted onto GRPO. Nothing to do with the importance weight.
#                     THIS is what run5 adds. PDF: docs/papers/2607.16850_*.pdf
# They are independent switches and run5 has BOTH on: --no-gepo is absent (expectation
# IS stays on) and --gepo-entropy is set. Exactly ONE variable moves against run4.
#
# THE RULE (Eq. 4, 7, 8, 9)
#   group entropy, per-token normalised:  H_g = -(sum_i sum_t log pi) / (sum_i T_i)
#   asymmetric shaping:                   A_hat = alpha_low  * A   if A>0 and H_g<H_low
#                                         A_hat = alpha_high * A   if A<0 and H_g>H_high
#                                         A_hat = A                otherwise
#   adaptive thresholds:                  H_low_hat  = mu - beta_low  * sigma
#                                         H_high_hat = mu + beta_high * sigma
#   EMA:                                  H_* <- (1-gamma) H_* + gamma H_*_hat
# alpha_high(0.2) < alpha_low(0.5) IS A REQUIREMENT, not a preference: the paper
# attributes LENGTH COLLAPSE -- "the model dramatically shortens its responses to reduce
# per-token uncertainty" -- to penalising negative advantages hard in the low-entropy
# regime. gepo_brevity.py REFUSES the inverted pair before it loads the model.
#
# WHY THIS RUN CAN SUCCEED WHERE run4 DID NOT
# ------------------------------------------------------------------------------
# run4 was not damaging, it was INERT. Final suite means: armJ 0.9083, ck60 0.9010,
# ck218 0.8923 -- ck60 (-0.73pp) is the best GEPO arm ever measured, and there is no
# dose ordering (ck218 vs ck60: GPQA p=0.7201, LCB p=0.8238). And no brevity either:
# total tokens armJ 1557577 -> ck60 1610920 (+3.4%) -> ck218 1612627 (+3.5%), with the
# LCB median completion FLAT at 12725/12703/12716 on the only lambda=0.7 tier. The
# rebalanced pool bought safety by making the pressure ineffective -- only 128/849 rows
# (15%) carry lambda>0. Entropy shaping acts on the ADVANTAGE, not on the reward, so it
# is not diluted by the tier mix the way a length penalty is.
#
# WHAT IS UNCHANGED FROM run4 (so the comparison is clean)
#   pool (849 rows, same sha), reward v2, lambda 0.7, budget 12288, LR 1e-6, temp 0.9,
#   beta 0.02, r/alpha 32/64, dropout 0.05, ROUTER_LORA=0 (run3 tested widening scope to
#   the 40 MoE routers and it LOST -- lowest suite mean of all four arms).
#
# WHAT CHANGES
#   1. --gepo-entropy with the paper's five defaults.
#   2. G 8 -> 12, and the pool SUBSAMPLED to 250 rows, to fit a 2-day budget. Both of
#      those are forced by the wall clock; see the two blocks below.
#
# WALL CLOCK: ~43 h. THE BUDGET IS THE DESIGN CONSTRAINT, NOT AN OUTCOME.
#   The measured rate is ~50.9 s PER ROLLOUT (run4: 6792 rollouts / ~96 h). Per-rollout
#   is the right unit and it transfers, because the pool, tier mix, length cap, model and
#   backend are identical to run4 -- only samples-per-prompt and row count move.
#   [[feedback_tokens_per_hour_does_not_transfer_across_pools]]
#     250 problems x G=12 = 3000 rollouts x 50.9 s = ~42.4 h, + ~0.8 h smoke = ~43 h.
#     steps = 3000 / (pdb 1 x accum 12 x world 2 = 24) x num_iterations 2 = 250
#   The full 849-row pool at G=16 would be 13584 rollouts = ~192 h (8 days). Rejected.
#
# THERE IS NO FASTER PATH. Both obvious levers are already closed:
#   * vLLM generation would be the 3-5x win, and use_vllm=False is NOT an oversight.
#     vLLM 0.20.2 mis-implements Qwen3_5MoeForCausalLM: the 2026-08-23 smoke generated
#     token soup, scored 0.0 on all 32 rollouts and logged grad_norm=0 on every step.
#     See the block above use_vllm=False in gepo_brevity.py.
#   * num_iterations=1 would halve optimizer steps but not generation, which is the
#     bottleneck -- and it is load-bearing anyway: num_iterations>1 is what makes
#     old_per_token_logps exist off the vLLM path, and the importance weight is DEFINED
#     against logp_theta_old. Setting it to 1 removes the quantity GEPO needs.
#   Lowering MAXCOMP below 12288 would buy time by CENSORING the tier under study --
#   it turns the measurement into a length meter. Not a lever, a bias.
#     [[feedback_cap_asymmetry_turns_a_bench_into_a_length_meter]]
#
# WHY G=12 AND NOT 16. At a fixed wall clock the rollout budget is fixed, so
# problems = rollouts/G: G trades DIRECTLY against how many distinct problems the run
# sees, and each problem is exactly one group.
#     G=16 -> 187 problems, 30 lcb_exec/T   (the paper's K)
#     G=12 -> 250 problems, 41 lcb_exec/T   <- chosen
#     G=8  -> 375 problems, 61 lcb_exec/T   (run4's K)
# Group-entropy variance is O(1/K), which argues up; but the objective lives on the
# 16% lcb_exec/T tier and run4's ck218 saw ~66 of those problems while showing nothing,
# so distinct coverage of THAT tier is the scarcer resource. G=12 keeps K 50% above
# run4 while getting 37% more LCB problems than G=16 would.
#
# POOL: SUBSAMPLED, NOT RESHAPED. --limit 250 takes the first 250 rows of the pool,
# which is INTERLEAVED, not grouped by tier (measured: 562 adjacent-tier alternations
# across 849 rows), so the tier mix survives the cut:
#     full 849: mbpp_exec/N 371  mc_letter/N 250  lcb_exec/T 128 (15.1%)  mbpp_exec/T 100
#     first 250: mbpp_exec/N 100  mc_letter/N  71  lcb_exec/T  41 (16.4%)  mbpp_exec/T 38
# The heterogeneity that made run4 SAFE is therefore preserved -- this is a smaller
# sample of the same pool, not the LCB-heavy pool of run1-3 that lost capability.
# gepo_brevity.py LOGS the realised per-tier counts after limiting: read them back, do
# not trust this comment. [[feedback_verify_eval_basis_by_hash_before_tabulating]]
# The pool FILE is unchanged (sha256 687eceed...), so run4 and run5 share a basis; the
# row count is recorded in run_meta.json.
#
# DECISION GATE AT ~STEP 60 (~10 h). run4's mistake was being unreadable until the eval.
#   run5 prints, EVERY step, after the 25-step warmup:
#     >>> GEPO_ENTROPY H_mean=.. H_std=.. lo=.. hi=.. groups=2 pos_att=.. neg_att=..
#         any=.. lcb_exec/T:1/1@H1.05 mbpp_exec/N:0/1@H1.31 ...
#   Read the PER-TIER field, never `any=`: 128/849 rows carry lambda>0, so a pool-wide
#   rate cannot see whether the tier under study was touched at all.
#   [[feedback_a_pool_wide_metric_cannot_see_a_minority_tier_objective]]
#   Kill and rethink if, by step ~60: lcb_exec/T shows 0 shaped groups, OR the reward
#   log's per-tier tok= for lcb_exec/T is flat within noise against its first 15 steps.
#
# THE FAILURE MODE THAT LOOKS LIKE SUCCESS. The paper's own pathology -- length collapse
# -- would read as a WIN on a brevity metric. Shorter is only good if it is also correct,
# so the run5 verdict is a LENGTH x SCORE cross-tab, never a length subtotal:
#   * per-tier tok= AND per-tier pass= must be read together, from the same steps;
#   * at eval, the 9-bench suite at the qwen_suite sampler, and the LCB cap-hit count
#     alongside the score -- a capped arm vs an uncapped one turns the bench into a
#     length meter. [[feedback_cap_asymmetry_turns_a_bench_into_a_length_meter]]
#   A big token drop with ANY score loss is the paper's failure, not our objective.
#
# GATES, in order, before the 192 h leg:
#   1. artifact contamination gate on the mixed pool, on THIS host
#   2. gate_learned      -- non-zero grad_norm AND reward_std
#   3. REPLAY_TIER_ALIVE -- all FOUR tiers produce a non-zero reward at the FULL budget
#   4. ENTROPY_GATE      -- NEW. The smoke must SEED the thresholds and shape a non-zero
#      fraction of rollouts on at least one step. The smoke runs at warmup=1 and
#      silent-limit=0 so it can fire at all inside ~6 steps. A green smoke on a rule that
#      never once ran is exactly the run4 trap.
#   Batteries (run these on the build host first, they need no GPU):
#     python scripts/test_gepo_entropy.py        -> GEPO_ENTROPY_GATE pass=22 fail=0
#     python scripts/test_gepo_entropy_args.py   -> GEPO_ENTROPY_ARGS_GATE pass=23 fail=0
#     bash   scripts/test_gepo_entropy_gate.sh   -> ENTROPY_SMOKE_GATE pass=7 fail=0
#
# SAVE_STEPS=12 -> ~21 checkpoints over 250 steps at ~537 MB each = ~11 GB. run4's
# SAVE_STEPS=2 produced ~212 saves / ~114 GB; do not repeat that.
set -uo pipefail

REPO=/srv/ml/repos/omnimergekit

exec env \
  RUN=run5 \
  REWARD=v2 \
  POOL="$REPO/eval/replay/gepo_mixed_pool.jsonl" \
  POOL_LIMIT="${POOL_LIMIT:-250}" \
  REPLAY_DIFFICULTY="${REPLAY_DIFFICULTY:-}" \
  ROUTER_LORA=0 \
  MAXCOMP=12288 \
  BUDGET=12288 \
  LAMBDA=0.7 \
  G="${G:-12}" \
  ACCUM="${ACCUM:-12}" \
  LR=1e-6 \
  TEMPERATURE=0.9 \
  BETA=0.02 \
  EPOCHS=1 \
  SAVE_STEPS="${SAVE_STEPS:-12}" \
  GEPO_ENTROPY=1 \
  ENT_ALPHA_LOW=0.5 \
  ENT_ALPHA_HIGH=0.2 \
  ENT_BETA_LOW=0.2 \
  ENT_BETA_HIGH=0.3 \
  ENT_GAMMA=0.01 \
  ENT_WARMUP="${ENT_WARMUP:-15}" \
  ENT_SILENT_LIMIT=25 \
  bash "$(dirname "$0")/r9_gepo_hf.sh" "$@"
