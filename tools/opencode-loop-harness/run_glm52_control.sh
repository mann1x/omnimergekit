#!/usr/bin/env bash
# run_glm52_control.sh — HARNESS FALSIFICATION ARM (2026-07-25).
#
# Question: is snake-adversarial completable at all through this harness, or does the
# harness condemn every model that runs it? snake-adversarial is the one task family where
# BOTH 128e (9/30 COMPLETED) and dense gemma-31b (3/114 COMPLETED) "fail", which is not a
# credible model result for an unpruned 31B across three independent backends.
#
# Control model: glm-5.2:cloud via ollama (a frontier model, zero GPU — does not touch
# GPU0/GPU1 or the :8240 opencoti prod supervisor). If a frontier model also gets stamped
# a failure verdict here, the harness is the problem, not the models.
#
# Held IDENTICAL to the g31-*-t1 arms so the comparison is apples-to-apples:
#   TASK_ID=snake-adversarial · PER_TURN_TIMEOUT=600 · same 8-turn adversarial script.
#
# Every session writes conversation.md (compact_session.py -> dump_transcript.py) in
# addition to the raw wirelog, so the whole exchange is on disk for audit.
set -uo pipefail
ROOT="${OPENCODE_CAPTURE_ROOT:-/mnt/sdc/ml/opencode_capture}"
cd "$ROOT"
N="${N:-3}"

for i in $(seq 1 "$N"); do
    echo "======== glm-5.2:cloud control session $i/$N $(date -u +%FT%TZ) ========"
    MODEL_PORT=11434 MODEL_NAME="glm-5.2:cloud" MODEL_LABEL=glm52-cloud-t1 \
    TASK_ID=snake-adversarial PROXY_PORT=8097 PER_TURN_TIMEOUT=600 \
        bash run_session_multiturn.sh
done
echo ">>> GLM52_CONTROL_DONE $(date -u +%FT%TZ)"
