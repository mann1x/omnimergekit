#!/usr/bin/env bash
# R9 run4a -- the v2 reward ALONE. One variable against run2.
#
# WHY THIS EXISTS SEPARATELY FROM run4
# ------------------------------------
# run4 changes two things at once: the reward (v1 -> v2) and the addition of a 64-row
# lambda=0 capability-replay tier. That was defensible when both rested on measured
# ground. As of 2026-08-28 only one of them does.
#
# STILL SOLID -- the reward diagnosis.
#   v1 was r = (1 - 0.7*min(ntok/12288,1)) if tests pass else 0. GRPO normalises
#   advantage by the WITHIN-GROUP std, so what matters is a term's share of that
#   variance, and length owned 11.3% of it -- decaying to 0.8% as lengths converged.
#   ~89% of the gradient said "be correct". That is a sufficient explanation for run1-3
#   moving mean length ~4% over a full epoch. v2 scores length group-relatively on a
#   robust scale-free statistic and gate-measures 86.5%, holding at 86.5% even at 10x
#   shorter lengths. Clean hypothesis, untested.
#
# NO LONGER SOLID -- the replay rationale.
#   run4's replay tier was designed to defend GPQA and HumanEval on the theory that
#   brevity pressure was shortening the reasoning those benches need. The served eval
#   store refutes the mechanism: completion length is FLAT across the whole dose series
#   while the scores fall.
#       GPQA        armJ 83.33 -> gepo2 75.76   completion p50 804 -> 792
#       HumanEval   armJ 98.17 -> gepo2 94.51   completion p50 220 -> 223
#   Whatever erodes those benches, it is not the model thinking less. Replay may still
#   help as a KL anchor over those domains (beta=0.02 keeps grpo_trainer.py:3121 active
#   regardless of advantage), but that is a weaker and less specific claim than the one
#   the tier was built for -- and mbpp_exec measured SATURATED (8/8 pass), so most of
#   that tier is anchor rather than reward signal.
#   [[project_gpqa_cot_needs_16k_on_armj]]
#
# THE READ THIS BUYS
#   run4a differs from run2 in EXACTLY ONE FIELD: --reward v2. Same 128 LCB problems,
#   same scope, same G/ACCUM/LR/BETA/EPOCHS, same 12288 budget, same lambda 0.7, same
#   lora_dropout (ROUTER_LORA=0 restores it). So run4a-vs-run2 isolates the reward, and
#   a null result MEANS something. run4-as-designed moves two variables and adds 64
#   problems, where a null would be ambiguous and a win unattributable.
#
# WALL CLOCK: 128 problems, i.e. run2's shape -- ~20h, versus 25-30h for run4.
#
# IF THIS PRODUCES BREVITY, run4 (reward v2 + replay, no-think, difficulty-filtered) is
# the natural follow-up and is already wired -- scripts/r9_gepo_run4.sh. If it does not,
# the replay tier was never the reason and run4 would have been 25-30h spent on the
# wrong variable.
set -uo pipefail

exec env \
  RUN=run4a \
  REWARD=v2 \
  ROUTER_LORA=0 \
  POOL_LIMIT=128 \
  MAXCOMP=12288 \
  BUDGET=12288 \
  LAMBDA=0.7 \
  G=8 \
  ACCUM=16 \
  LR=5e-6 \
  BETA=0.02 \
  EPOCHS=1 \
  SAVE_STEPS=8 \
  bash "$(dirname "$0")/r9_gepo_hf.sh" "$@"
