#!/usr/bin/env bash
# smoke_iq3m_noimat.sh <GPU> <PORT> — decisive 3-prompt smoke of the noimat IQ3_M
# rebuild. Serves with the SAME flags the sweep used, sends 3 short code prompts
# at greedy, prints length + loop-signal + head of each reply. Kills server by
# verified PID. ~2 min. Verdict: clean code -> noimat rescues IQ3_M; garbage ->
# IQ3_M structurally broken for the 98e prune (drop it).
set -uo pipefail
GPU="${1:?usage: smoke_iq3m_noimat.sh <GPU> <PORT>}"
PORT="${2:?usage: smoke_iq3m_noimat.sh <GPU> <PORT>}"
BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin/llama-server
GGUF=/mnt/sdc/ml/v8_tier_sweep/gguf/v8-IQ3_M-noimat.gguf
PY=/root/anaconda3/envs/omnimergekit/bin/python
L(){ echo "[smoke $(date -u +%T)] $*"; }

[ -f "$GGUF" ] || { L "FATAL gguf missing $GGUF"; exit 1; }
L "serving noimat IQ3_M on GPU$GPU:$PORT"
CUDA_VISIBLE_DEVICES="$GPU" "$BIN" -m "$GGUF" --port "$PORT" -c 8192 -ngl 99 \
  --no-warmup --jinja --reasoning-format deepseek --reasoning-budget 8192 \
  > /mnt/sdc/ml/v8_tier_sweep/work/iq3m_noimat_server.log 2>&1 &
SRV=$!
disown
trap 'kill "$SRV" 2>/dev/null || true' EXIT
# wait for ready (verified PID)
for i in $(seq 1 60); do
  kill -0 "$SRV" 2>/dev/null || { L "FATAL server died early"; tail -15 /mnt/sdc/ml/v8_tier_sweep/work/iq3m_noimat_server.log; exit 1; }
  curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && { L "ready"; break; }
  sleep 3
done

prompts=(
  "Write a Python function is_prime(n) that returns True if n is prime."
  "Write a Python function reverse_string(s) that returns the reversed string."
  "Implement binary search in Python: def bsearch(arr, target) -> int (index or -1)."
)
verdict_clean=1
for i in 0 1 2; do
  p="${prompts[$i]}"
  resp=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$("$PY" -c "import json,sys;print(json.dumps({'messages':[{'role':'user','content':sys.argv[1]}],'temperature':0.0,'max_tokens':512}))" "$p")" 2>/dev/null)
  txt=$("$PY" -c "import json,sys;d=json.loads(sys.stdin.read());print(d['choices'][0]['message'].get('content') or d['choices'][0]['message'].get('reasoning_content') or '')" <<<"$resp" 2>/dev/null)
  # loop signal
  lp=$("$PY" -c "
import sys
s=sys.argv[1]
w=12
best=1
seen={}
for i in range(0,max(1,len(s)-w),3):
    sub=s[i:i+w]; seen[sub]=seen.get(sub,0)+1; best=max(best,seen[sub])
print(best)
" "$txt" 2>/dev/null)
  hasdef=$(printf '%s' "$txt" | grep -c "def " || true)
  head=$(printf '%s' "$txt" | head -c 160 | tr '\n' ' ')
  L "[q$i] len=${#txt} loop=$lp def_count=$hasdef head=$head"
  [ "${#txt}" -gt 4000 ] && verdict_clean=0
  [ "${lp:-0}" -gt 8 ] && verdict_clean=0
  [ "${hasdef:-0}" -lt 1 ] && verdict_clean=0
done
kill "$SRV" 2>/dev/null || true
if [ "$verdict_clean" -eq 1 ]; then
  L "VERDICT: CLEAN — noimat rescues IQ3_M (proceed to full HE+/MPE)"
else
  L "VERDICT: DEGENERATE — IQ3_M structurally broken for v8 (drop tier)"
fi
