#!/usr/bin/env bash
# smoke_gguf.sh <gguf> <port> <gpu> [max_tokens] — generic termination smoke.
# Serves one GGUF with the EXACT sweep flags (--jinja --reasoning off) and probes
# 5 trivial prompts. Healthy = short answer + finish=stop. Ruminator = hits the
# max_tokens cap (finish=length) OR the server 500s on malformed channel salad.
set -uo pipefail
GGUF="${1:?gguf}"; PORT="${2:?port}"; GPU="${3:?gpu}"; MAXTOK="${4:-2048}"
LLAMA=/opt/llama.cpp/build/bin/llama-server
PY=/srv/ml/envs/envs/omnimergekit/bin/python
[ -f "$GGUF" ] || { echo "[FATAL] missing $GGUF"; exit 1; }
echo "==== smoke $(basename "$GGUF")  GPU$GPU port$PORT maxtok=$MAXTOK  $(date -u) ===="
CUDA_VISIBLE_DEVICES=$GPU "$LLAMA" -m "$GGUF" --port "$PORT" -c 32768 -ngl 99 \
    --parallel 1 --no-warmup --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --reasoning off >/tmp/smoke_$PORT.log 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null' EXIT
for i in $(seq 1 90); do curl -s -m 3 "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok"\|"status":"ok"' && break; sleep 2; done
PROMPTS=(
  "What is the capital of France? Answer with one word."
  "Write a Python function add(a, b) that returns a+b. Only the code."
  "What is 7 times 6? Give only the number."
  "Reverse the string 'hello'. Give only the result."
  "Name three primary colors, comma-separated."
)
rum=0; ok=0
for p in "${PROMPTS[@]}"; do
  body=$("$PY" -c 'import json,sys;print(json.dumps({"model":"s","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[2]),"temperature":0}))' "$p" "$MAXTOK")
  resp=$(curl -s -m 120 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
  verdict=$("$PY" - "$resp" <<'PYEOF'
import json,sys
r=sys.argv[1]
try:
    d=json.loads(r)
    if "error" in d:
        print(f"    500-ERR len={len(r):6d} msg={str(d['error'].get('message',''))[:60]!r}")
        sys.exit(7)
    ch=d["choices"][0]; m=ch["message"]
    c=(m.get("content") or "") or (m.get("reasoning_content") or ""); fr=ch.get("finish_reason")
    flag = "RUMINATE" if fr=="length" else ("STOP-ok" if fr=="stop" else fr)
    print(f"    {flag:9s} len={len(c):6d} finish={fr!s:8s} head={c[:55]!r}")
    sys.exit(0 if fr=="stop" else 7)
except SystemExit: raise
except Exception as e:
    print(f"    PARSE-ERR {e} raw0={r[:60]!r}")
    sys.exit(7)
PYEOF
)
  echo "$verdict"
  if echo "$verdict" | grep -q "STOP-ok"; then ok=$((ok+1)); else rum=$((rum+1)); fi
done
echo "==== RESULT: $ok/5 STOP, $rum/5 BROKEN ===="
