#!/usr/bin/env bash
# gate_shipped_v8.sh <GGUF> <GPU> <PORT> <OUT> <LABEL>
# Loop gate under the ACTUAL shipped v8 deployment serving config (per GGUF card How-to-Use):
#   --reasoning-budget 8192   (card-REQUIRED value; bounds thinking-channel rumination —
#                              NOT the cross-variant ledger's 12288)
#   sampler = --temp 0.6 --min-p 0.05 --repeat-penalty 1.1, with top_p/top_k at llama-server
#             DEFAULTS (top_p 0.95 / top_k 40). (Earlier wrong gate used top_p 1.0/top_k 0 +
#             a non-shipped temp-1.0 arm.)
# ctx kept high (-c 131072) to fit the ~20k-token agentic solar fixture (card: "raise num_ctx
# for very long tasks"); MAXTOK 16384 runaway ceiling (comparable to the prior gates).
# 48 seeds 1000-1047. llama.cpp-latest. PID-kill only.
set -uo pipefail
GGUF="$1"; GPU="$2"; PORT="$3"; OUT="$4"; LABEL="$5"
AL=/srv/ml/agentic_loop; cd "$AL"
LS=/mnt/sdc/ml/llama.cpp-latest/build/bin/llama-server
PY=/root/anaconda3/envs/omnimergekit/bin/python
MATRIX="${MATRIX:-matrix_shipped_v8_real.json}"
SEEDS=$(seq -s, 1000 1047)
MAXTOK=16384
BUDGET=8192
ts(){ date '+%T %Z'; }
for f in "$LS" "$GGUF" "$MATRIX" fixtures/solar_build_start.json; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
mkdir -p logs results
echo "==================== gate_shipped_v8 $LABEL $(ts) ===================="
echo "[$(ts)] $LABEL server GPU$GPU:$PORT  --reasoning-budget $BUDGET (SHIPPED)  max=$MAXTOK  matrix=$MATRIX"
CUDA_VISIBLE_DEVICES=$GPU nohup "$LS" -m "$GGUF" --host 127.0.0.1 --port "$PORT" \
  --alias rp --jinja -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 131072 --parallel 1 --no-warmup \
  --reasoning-format deepseek --reasoning-budget $BUDGET \
  > "logs/shipped_srv_${PORT}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
OK=0
for i in $(seq 1 300); do
  curl -fsS -m 10 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' \
    >/dev/null 2>&1 && { echo "[$(ts)] :$PORT SERVING ~$((i*2))s"; OK=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "[$(ts)] SERVER DIED"; tail -20 "logs/shipped_srv_${PORT}.log"; exit 3; }
  sleep 2
done
[ $OK = 1 ] || { echo "server never came up"; exit 3; }
echo "[$(ts)] replay 48 seeds ($LABEL, SHIPPED config)"
"$PY" replay_harness.py --fixture fixtures/solar_build_start.json --server "http://127.0.0.1:${PORT}" \
  --matrix "$MATRIX" --seed-list "$SEEDS" --max-tokens "$MAXTOK" \
  --out "$OUT" --timeout 1800 > "logs/shipped_replay_${LABEL}.log" 2>&1
kill $SRV 2>/dev/null; trap - EXIT; sleep 2
echo "==== $LABEL: any FAIL=True (loopers under SHIPPED config) ===="
grep -E "FAIL=True" "logs/shipped_replay_${LABEL}.log" || echo "  (none — 0 loopers)"
echo "==== $LABEL arm summary ===="
"$PY" - "$OUT" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d["results"]:
    loops = r.get("loops", round(r.get("loop_rate", 0) * r["seeds"]))
    print("  %-12s fails=%d/%d  loops=%s  fail_rate=%.1f%%"
          % (r["config"], r["fails"], r["seeds"], loops, 100 * r["fail_rate"]))
PYEOF
echo "[$(ts)] === $LABEL DONE (SHIPPED config: budget=$BUDGET, temp0.6/tp0.95/tk40/minp0.05/rep1.1) ==="
echo "###### SHIPPED_GATE_DONE $(ts) ######"
