#!/bin/bash
# pod_v4_q6k_rerun_failed.sh — re-run the 5 phases the 2026-05-22 chain broke
# on pod 37268930. std_he (83.54% pass@1) survives intact.
#
# Root causes patched in the canonical pod_v4_q6k_eval_chain.sh:
#   1. lm-eval local-completions default max_length=2048 truncated MBPP
#      3-shot prompts and forced "[invalid]" on every GPQA prompt
#      (gen_toks 4096-8192 > max_length-prompt). Fix: max_length=32768.
#   2. MTP server CUDA OOM at -c 65536 --parallel 2 on a 24 GB 3090
#      (Q6_K 21 GB + draft buffer + 2-slot KV > 24 GB). Fix: MTP path
#      now boots at -c 16384 --parallel 1.
#
# This wrapper:
#   - purges sqlite caches that hold cached "[invalid]"/empty completions
#   - purges stale result dirs (samples_*.jsonl, results_*.json, throughput.json)
#   - sources nothing — it just re-execs the chain with PHASES restricted
#
# Preserves: std_he/* (valid 83.54% pass@1).

set -uo pipefail
WORK=/workspace
EVAL_ROOT=$WORK/eval_results

FAILED_PHASES="std_mbpp,std_gpqa,mtp_he,mtp_mbpp,mtp_gpqa"

echo "=================================================================="
echo "  pod_v4_q6k_rerun_failed.sh"
echo "  started: $(date)"
echo "  re-running phases: $FAILED_PHASES"
echo "=================================================================="

# Purge stale caches + result dirs
for ph in std_mbpp std_gpqa mtp_he mtp_mbpp mtp_gpqa; do
    d="$EVAL_ROOT/$ph"
    if [ -d "$d" ]; then
        echo "[purge] $d ($(du -sh $d 2>/dev/null | cut -f1))"
        rm -rf "$d"
    else
        echo "[purge] $d (absent, skip)"
    fi
done

# Sanity: std_he must survive
if [ -f "$EVAL_ROOT/std_he/rescored_clean.json" ]; then
    pa=$(python3 -c "import json; d=json.load(open('$EVAL_ROOT/std_he/rescored_clean.json')); print(d.get('pass@1'))" 2>/dev/null)
    echo "[keep] std_he/rescored_clean.json pass@1=$pa  ✓"
else
    echo "[WARN] std_he/rescored_clean.json missing — chain may rebuild it"
fi

# Re-exec chain with PHASES filter. Inherits HF_TOKEN / env from caller.
echo
echo "[$(date +%H:%M:%S)] launching chain with PHASES=$FAILED_PHASES"
PHASES="$FAILED_PHASES" exec bash /workspace/scripts/pod_v4_q6k_eval_chain.sh
