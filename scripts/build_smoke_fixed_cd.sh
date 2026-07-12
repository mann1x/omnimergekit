#!/usr/bin/env bash
# build_smoke_fixed_cd.sh — PROOF of the attn-protect fix. Rebuild v7-coder
# CD-Q4_K_M from F16 using the FIXED map (attn_q/attn_output now Q5_K), then
# termination-smoke it on the dedicated CD GPU0. Healthy = short answers that
# STOP; broken = ruminates to the cap.
set -uo pipefail
GPU=0; PORT=8261
LLAMA=/opt/llama.cpp/build/bin/llama-server
QUANT=/opt/llama.cpp/build/bin/llama-quantize
PY=/srv/ml/envs/envs/omnimergekit/bin/python
GD=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF
F16="$GD/gemma-4-A4B-98e-v7-coder-it-F16.gguf"
IMAT="$GD/imatrix.dat"
MAP=/srv/ml/scripts/cd_maps_fixed_v7coder/tensor_types_CD-Q4_K_M.txt
OUT=/mnt/sdc/ml/quant_sweep_gguf/gemma-4-A4B-98e-v7-coder-it-CD-Q4_K_M-FIXED.gguf
LOG=/srv/ml/logs/build_smoke_fixed_cd.txt
: > "$LOG"

for f in "$F16" "$IMAT" "$MAP"; do [ -f "$f" ] || { echo "[FATAL] missing $f" | tee -a "$LOG"; exit 1; }; done

echo "==== [1/2] rebuild CD-Q4_K_M-FIXED from F16 (CPU) $(date -u) ====" | tee -a "$LOG"
"$QUANT" --imatrix "$IMAT" --tensor-type-file "$MAP" "$F16" "$OUT" Q4_K_M >>"$LOG" 2>&1
rc=$?
[ $rc -eq 0 ] && [ -f "$OUT" ] || { echo "[FATAL] quantize rc=$rc" | tee -a "$LOG"; tail -5 "$LOG"; exit 1; }
echo "  built $(du -h "$OUT" | cut -f1) -> $OUT" | tee -a "$LOG"

echo "==== [2/2] termination smoke on GPU$GPU $(date -u) ====" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU "$LLAMA" -m "$OUT" --port $PORT -c 32768 -ngl 99 \
    --parallel 1 --no-warmup --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --reasoning off >/tmp/fixed_cd_$PORT.log 2>&1 &
srv=$!
for i in $(seq 1 90); do curl -s -m 3 http://localhost:$PORT/health 2>/dev/null | grep -q '"ok"\|"status":"ok"' && break; sleep 2; done
PROMPTS=(
  "What is the capital of France? Answer with one word."
  "Write a Python function add(a, b) that returns a+b. Only the code."
  "What is 7 times 6? Give only the number."
  "Reverse the string 'hello'. Give only the result."
  "Name three primary colors, comma-separated."
)
for p in "${PROMPTS[@]}"; do
  body=$("$PY" -c 'import json,sys;print(json.dumps({"model":"s","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":4096,"temperature":0}))' "$p")
  resp=$(curl -s -m 300 http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$body")
  "$PY" - "$resp" <<'PYEOF' | tee -a "$LOG"
import json,sys
try:
    d=json.loads(sys.argv[1]); ch=d["choices"][0]; m=ch["message"]
    c=(m.get("content") or "") or (m.get("reasoning_content") or ""); fr=ch.get("finish_reason")
    print(f"    len={len(c):6d} finish={fr:8s} head={c[:55]!r}")
except Exception as e:
    print(f"    PARSE-ERR {e}")
PYEOF
done
kill $srv 2>/dev/null; wait $srv 2>/dev/null
echo "[done] $(date -u)" | tee -a "$LOG"
