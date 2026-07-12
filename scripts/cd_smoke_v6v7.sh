#!/usr/bin/env bash
# cd_smoke_v6v7.sh — settle "recipe vs weights": probe v6 vs v7 CD-Q4_K_M
# (byte-identical quant recipe) for termination. Serves with the EXACT sweep
# flags (--jinja --reasoning off) so behavior matches the HE+ eval.
# Dedicated CD GPU0. Healthy = short answer + finish_reason=stop; ruminator =
# hits the max_tokens cap (finish_reason=length).
set -uo pipefail
GPU=0; PORT=8260
LLAMA=/opt/llama.cpp/build/bin/llama-server
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OUT=/srv/ml/logs/cd_smoke_v6v7.txt
: > "$OUT"

declare -A G=(
  [v6-CD-Q4_K_M]=/mnt/sdc/ml/v6_cd_compare/gemma-4-A4B-98e-v6-coder-it-CD-Q4_K_M.gguf
  [v7-CD-Q4_K_M]=/mnt/sdc/ml/quant_sweep_gguf/gemma-4-A4B-98e-v7-coder-it-CD-Q4_K_M.gguf
)
PROMPTS=(
  "What is the capital of France? Answer with one word."
  "Write a Python function add(a, b) that returns a+b. Only the code."
  "What is 7 times 6? Give only the number."
  "Reverse the string 'hello'. Give only the result."
  "Name three primary colors, comma-separated."
)

probe() {  # port -> prints per-prompt len + finish_reason
  local port="$1" p body resp
  for p in "${PROMPTS[@]}"; do
    body=$("$PY" -c 'import json,sys;print(json.dumps({"model":"s","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":4096,"temperature":0}))' "$p")
    resp=$(curl -s -m 300 http://localhost:"$port"/v1/chat/completions -H 'Content-Type: application/json' -d "$body")
    "$PY" - "$resp" <<'PYEOF'
import json,sys
try:
    d=json.loads(sys.argv[1]); ch=d["choices"][0]; m=ch["message"]
    c=(m.get("content") or "") or (m.get("reasoning_content") or ""); fr=ch.get("finish_reason")
    print(f"    len={len(c):6d} finish={fr:8s} head={c[:50]!r}")
except Exception as e:
    print(f"    PARSE-ERR {e}")
PYEOF
  done
}

for name in v6-CD-Q4_K_M v7-CD-Q4_K_M; do
  gguf="${G[$name]}"
  echo "==== $name  $(date -u) ====" | tee -a "$OUT"
  [ -f "$gguf" ] || { echo "  MISSING $gguf" | tee -a "$OUT"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU "$LLAMA" -m "$gguf" --port $PORT -c 32768 -ngl 99 \
      --parallel 1 --no-warmup --cache-type-k q8_0 --cache-type-v q8_0 \
      --jinja --reasoning off >/tmp/cd_smoke_$PORT.log 2>&1 &
  srv=$!
  for i in $(seq 1 90); do curl -s -m 3 http://localhost:$PORT/health 2>/dev/null | grep -q '"ok"\|"status":"ok"' && break; sleep 2; done
  probe $PORT | tee -a "$OUT"
  kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 3
done
echo "[cd_smoke] DONE $(date -u)" | tee -a "$OUT"
