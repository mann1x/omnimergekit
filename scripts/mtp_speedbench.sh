#!/usr/bin/env bash
# T178 MTP speed bench: launch one llama-server config, run identical greedy
# completions, report server-side predicted tok/s + (for MTP) draft acceptance.
# Usage: mtp_speedbench.sh <base|mtp> <port>
set -u
MODE="${1:?mode base|mtp}"; PORT="${2:?port}"
NEW=/srv/ml/repos/llama.cpp-latest/build/bin
GGUF=/mnt/sdc/ml/gguf/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
LOG=/srv/ml/logs/t176/mtp_bench_${MODE}.log
SPEC=()
NMAX=${3:-3}; [ "$MODE" = "mtp" ] && SPEC=(--spec-type draft-mtp --spec-draft-n-max $NMAX)
pkill -f "llama-server.*${PORT}" 2>/dev/null; sleep 2
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$NEW nohup "$NEW/llama-server" -m "$GGUF" \
  --port "$PORT" -c 8192 -ngl 99 -ctk q8_0 -ctv q8_0 "${SPEC[@]}" \
  > "$LOG" 2>&1 < /dev/null &
SRV=$!
for i in $(seq 1 120); do
  curl -s "localhost:${PORT}/health" 2>/dev/null | grep -q ok && break
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -15 "$LOG"; exit 1; }
  sleep 2
done
curl -s "localhost:${PORT}/health" 2>/dev/null | grep -q ok || { echo "NO HEALTH"; tail -15 "$LOG"; exit 1; }
echo "[$MODE] server up on $PORT"
# how spec engaged
grep -iE "adding speculative implementation|no implementations specified|draft-mtp|LLAMA_CONTEXT_TYPE_MTP|will use checkpoints" "$LOG" | head -5
PROMPT='<start_of_turn>user\nWrite a complete Python implementation of merge sort with a detailed step-by-step explanation of how it works, including time complexity analysis.<end_of_turn>\n<start_of_turn>model\n'
TOTAL=0; N=0
for run in 1 2 3; do
  R=$(curl -s "localhost:${PORT}/completion" -H "Content-Type: application/json" \
    -d "{\"prompt\":\"${PROMPT}\",\"n_predict\":500,\"temperature\":0,\"cache_prompt\":false}")
  TPS=$(echo "$R" | python3 -c "import json,sys;d=json.load(sys.stdin);t=d['timings'];print('%.3f %d'%(t['predicted_per_second'],t['predicted_n']))")
  echo "  run$run: tok/s=$(echo $TPS|cut -d' ' -f1) predicted_n=$(echo $TPS|cut -d' ' -f2)"
  TOTAL=$(python3 -c "print($TOTAL + $(echo $TPS|cut -d' ' -f1))"); N=$((N+1))
done
echo "[$MODE] MEAN tok/s = $(python3 -c "print('%.2f'%($TOTAL/$N))")"
if [ "$MODE" = "mtp" ]; then
  echo "=== draft acceptance (server log) ==="
  grep -iE "accept|draft|n_drafted|n_accept|spec" "$LOG" | tail -8
fi
pkill -f "llama-server.*${PORT}" 2>/dev/null
echo "[$MODE] done"
