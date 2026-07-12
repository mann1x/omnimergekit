#!/usr/bin/env bash
# gate_loopers_all.sh — re-test ALL 6 known loopers on dern11-Q4 WITH the proper Gemma 4
# reasoning flags (--reasoning-format deepseek --reasoning-budget 12288) the prior gate
# omitted (why every loop logged think=0). 6 seeds x 6 arms = 36 gens:
#   base_t{1.0,0.9,0.8}  (NO min_p / NO rep_pen)  <- native config of 1011,1024,1026,1036
#   minp_t{1.0,0.9,0.8}  (min_p0.05 + rep1.1)     <- native config of 1038,1039
# max_tokens 16384 so the 12288 reasoning budget binds. Reasoning probe proves thinking
# is live. PID-kill only.
set -uo pipefail
AL=/srv/ml/agentic_loop; cd "$AL"
LS=/mnt/sdc/ml/llama.cpp-latest/build/bin/llama-server
PY=/root/anaconda3/envs/omnimergekit/bin/python
GGUF=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-dern11-it-Q4_K_M.gguf
GPU=0; PORT=8190
MATRIX=matrix_loopers.json
SEEDS="1011,1024,1026,1036,1038,1039"
MAXTOK=16384
OUT=results/dern11_loops6_reasoning.json
ts(){ date '+%T %Z'; }
for f in "$LS" "$GGUF" "$MATRIX" fixtures/solar_build_start.json; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
mkdir -p logs results
echo "==================== gate_loopers_all $(ts) ===================="
echo "[$(ts)] SERVER: --jinja -fa on -ctk/-ctv q8_0 -c 131072 --parallel 1 \\"
echo "                 --reasoning-format deepseek --reasoning-budget 12288   (max_tok=$MAXTOK)"
CUDA_VISIBLE_DEVICES=$GPU nohup "$LS" -m "$GGUF" --host 127.0.0.1 --port $PORT \
  --alias rp --jinja -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 131072 --parallel 1 --no-warmup \
  --reasoning-format deepseek --reasoning-budget 12288 \
  > "logs/loop6_srv_${PORT}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
OK=0
for i in $(seq 1 300); do
  curl -fsS -m 10 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' \
    >/dev/null 2>&1 && { echo "[$(ts)] :$PORT SERVING ~$((i*2))s"; OK=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "[$(ts)] SERVER DIED"; tail -20 "logs/loop6_srv_${PORT}.log"; exit 3; }
  sleep 2
done
[ $OK = 1 ] || { echo "server never came up"; exit 3; }

echo "[$(ts)] === REASONING PROBE (prove the parser is active) ==="
"$PY" - "$PORT" <<'PYEOF'
import sys, requests
port = sys.argv[1]
r = requests.post("http://127.0.0.1:%s/v1/chat/completions" % port,
    json={"messages": [{"role": "user",
          "content": "A train travels 60 km in 45 minutes. Speed in km/h? Reason it out."}],
          "temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05,
          "repeat_penalty": 1.1, "max_tokens": 2048}, timeout=300).json()
m = r["choices"][0]["message"]
rc = m.get("reasoning_content") or ""
ct = m.get("content") or ""
print("  reasoning_len =", len(rc), " content_len =", len(ct),
      " finish =", r["choices"][0].get("finish_reason"))
print("  THINKING ACTIVE:", "YES" if rc else "NO (reasoning_content empty!)")
PYEOF

echo "[$(ts)] === replay 6 loopers x 6 arms (max=$MAXTOK) ==="
"$PY" replay_harness.py --fixture fixtures/solar_build_start.json --server "http://127.0.0.1:${PORT}" \
  --matrix "$MATRIX" --seed-list "$SEEDS" --max-tokens "$MAXTOK" \
  --out "$OUT" --timeout 1800 > "logs/loop6_replay.log" 2>&1
kill $SRV 2>/dev/null; trap - EXIT; sleep 2
echo "==== PER-(arm,seed) verdicts ===="
grep -E "seed=(1011|1024|1026|1036|1038|1039)" "logs/loop6_replay.log" || echo "(no per-seed lines)"
echo "==== arm summary (fails out of 6 seeds) ===="
"$PY" - "$OUT" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d["results"]:
    loops = r.get("loops", round(r.get("loop_rate", 0) * r["seeds"]))
    print("  %-10s fails=%d/%d  loops=%s  fail_rate=%.0f%%"
          % (r["config"], r["fails"], r["seeds"], loops, 100 * r["fail_rate"]))
PYEOF
echo "[$(ts)] DONE"