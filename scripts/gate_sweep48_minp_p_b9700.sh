#!/usr/bin/env bash
# gate_sweep48_minp.sh <GGUF> <GPU> <PORT> <OUT> <LABEL>
# Full 48-seed (1000-1047) confirmation at vendor_minp_rep x temps {0.9, 0.8}, with the
# proper Gemma 4 reasoning flags (--reasoning-format deepseek --reasoning-budget 12288)
# and max_tokens 16384 (budget binds). Confirms the known-looper fix generalizes (0/48)
# and surfaces ANY newly-induced looper by name. PID-kill only.
set -uo pipefail
GGUF="$1"; GPU="$2"; PORT="$3"; OUT="$4"; LABEL="$5"
AL=/srv/ml/agentic_loop; cd "$AL"
LS=/mnt/sdc/ml/llama.cpp-b9700/build/bin/llama-server
PY=/root/anaconda3/envs/omnimergekit/bin/python
MATRIX="${MATRIX:-matrix_minp_2temp.json}"   # override via env: MATRIX=matrix_minp_2temp_b.json
SEEDS=$(seq -s, 1000 1047)
MAXTOK=16384
ts(){ date '+%T %Z'; }
for f in "$LS" "$GGUF" "$MATRIX" fixtures/solar_build_start.json; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
mkdir -p logs results
echo "==================== gate_sweep48_minp $LABEL $(ts) ===================="
echo "[$(ts)] $LABEL server GPU$GPU:$PORT  --reasoning-format deepseek --reasoning-budget 12288  max=$MAXTOK"
CUDA_VISIBLE_DEVICES=$GPU nohup "$LS" -m "$GGUF" --host 127.0.0.1 --port "$PORT" \
  --alias rp --jinja -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 131072 --parallel 1 --no-warmup \
  --reasoning-format deepseek --reasoning-budget 12288 \
  > "logs/minp48_srv_${PORT}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
OK=0
for i in $(seq 1 300); do
  curl -fsS -m 10 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' \
    >/dev/null 2>&1 && { echo "[$(ts)] :$PORT SERVING ~$((i*2))s"; OK=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "[$(ts)] SERVER DIED"; tail -20 "logs/minp48_srv_${PORT}.log"; exit 3; }
  sleep 2
done
[ $OK = 1 ] || { echo "server never came up"; exit 3; }
echo "[$(ts)] replay 48 seeds x {minp_t0.9, minp_t0.8} ($LABEL)"
"$PY" replay_harness.py --fixture fixtures/solar_build_start.json --server "http://127.0.0.1:${PORT}" \
  --matrix "$MATRIX" --seed-list "$SEEDS" --max-tokens "$MAXTOK" \
  --out "$OUT" --timeout 1800 > "logs/minp48_replay_${LABEL}.log" 2>&1
kill $SRV 2>/dev/null; trap - EXIT; sleep 2
echo "==== $LABEL: any FAIL=True (newly-induced loopers) ===="
grep -E "FAIL=True" "logs/minp48_replay_${LABEL}.log" || echo "  (none — 0 loopers)"
echo "==== $LABEL arm summary ===="
"$PY" - "$OUT" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d["results"]:
    loops = r.get("loops", round(r.get("loop_rate", 0) * r["seeds"]))
    print("  %-10s fails=%d/%d  loops=%s  fail_rate=%.1f%%"
          % (r["config"], r["fails"], r["seeds"], loops, 100 * r["fail_rate"]))
PYEOF
echo "[$(ts)] === $LABEL DONE ==="