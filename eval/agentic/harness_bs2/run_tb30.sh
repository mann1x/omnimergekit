#!/usr/bin/env bash
# TB30 — terminal-bench, 30 tasks, paired template A/B.
# armA = live google template (re-injects), armB = one-line fix.
# Arms run on separate GPUs concurrently: the ENDPOINT is loop_rate, which is
# contention-independent. NO wall-time claim is made from this cell (see the
# ABBA base cell for latency).
set -uo pipefail
R=/shared/dev/omnimergekit/eval/agentic
G=/srv/ml/models/gguf/google-a4b-128e/google_gemma-4-26B-A4B-it-Q4_K_M-eos106.gguf
TP=/mnt/sdc/v7rework/template_probe
DS=/mnt/sdc/agentbench/datasets/terminal-bench
OUT=/mnt/sdc/agentbench/tb30
L=/mnt/sdc/harness/logs
mkdir -p "$OUT" "$L"
bash "$R/run_arm_harbor.sh" armA "$G" "$TP/google_live_chat_template.jinja" "$DS" "$OUT" 0 8401 30 > "$L/tb30_armA.log" 2>&1 &
PA=$!
sleep 240   # let armA warm the shared docker image cache before armB pulls
bash "$R/run_arm_harbor.sh" armB "$G" "$TP/v7coder_chat_template.jinja"     "$DS" "$OUT" 1 8402 30 > "$L/tb30_armB.log" 2>&1 &
PB=$!
wait $PA; echo "TB30_armA_EXIT=$?"
wait $PB; echo "TB30_armB_EXIT=$?"
echo "TB30_CELL_DONE $(date -u +%FT%TZ)"
