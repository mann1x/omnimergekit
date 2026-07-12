#!/usr/bin/env bash
# gate_dern11_48seed.sh — wider-CI agentic loop gate for the DERN-Eq.11 candidate.
# dern11 Q4 ONLY, 48 seeds (1000-1047) x 2 sampler arms via one resident server:
#   vendor_base      temp1.0/top_p0.95/top_k64/min_p0   NO penalty   (the 1/12 arm, tighter CI)
#   vendor_minp_rep  temp1.0/top_p0.95/top_k64/min_p0.05 repeat_penalty1.1 (mitigated user sampler)
# Co-located on GPU1:8193 (shares with the noswap trade-check; dern11 eval on GPU0 stays clean).
# Loop detection is token/text based, so GPU contention affects wall-time only, not the verdict.
# PID-kill only (never bare-port).
set -uo pipefail
AL=/srv/ml/agentic_loop; cd "$AL"
LS=/mnt/sdc/ml/llama.cpp-latest/build/bin/llama-server
PY=/root/anaconda3/envs/omnimergekit/bin/python
GGUF=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-dern11-it-Q4_K_M.gguf
GPU=1; PORT=8193
MATRIX=matrix_gate_48_2arm.json
SEEDS=$(seq -s, 1000 1047)
MAXTOK=8192
ts(){ date '+%T %Z'; }
for f in "$LS" "$GGUF" "$MATRIX" fixtures/solar_build_start.json; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
echo "==================== gate_dern11_48seed $(ts) ===================="
echo "[$(ts)] launch dern11 server GPU$GPU:$PORT"
CUDA_VISIBLE_DEVICES=$GPU nohup "$LS" -m "$GGUF" --host 127.0.0.1 --port $PORT \
  --alias rp --jinja -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 131072 --parallel 1 --no-warmup \
  > "logs/gate48_srv_${PORT}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 240); do
  curl -fsS -m 10 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' \
    >/dev/null 2>&1 && { echo "[$(ts)] :$PORT SERVING ~$((i*2))s"; break; }
  sleep 2
done
echo "[$(ts)] replay 48 seeds x 2 arms (max_tok=$MAXTOK)"
"$PY" replay_harness.py --fixture fixtures/solar_build_start.json --server "http://127.0.0.1:${PORT}" \
  --matrix "$MATRIX" --seed-list "$SEEDS" --max-tokens "$MAXTOK" \
  --out results/dern11_48seed.json --timeout 1500 > "logs/gate48_replay.log" 2>&1
kill $SRV 2>/dev/null; trap - EXIT; sleep 2
echo "==== DERN11 48-SEED SUMMARY ($(ts)) ===="
"$PY" - <<'PYEOF'
import json
d=json.load(open("results/dern11_48seed.json"))
for r in d["results"]:
    loops=r.get("loops", round(r.get("loop_rate",0)*r["seeds"]))
    runs =r.get("runaways", round(r.get("runaway_rate",0)*r["seeds"]))
    print("%-18s fails=%d/%d  loops=%s  runaways=%s  fail_rate=%.1f%%"
          % (r["config"], r["fails"], r["seeds"], loops, runs, 100*r["fail_rate"]))
PYEOF
echo "[$(ts)] === 48-SEED GATE DONE ==="
