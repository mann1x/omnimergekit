#!/usr/bin/env bash
# Smoke-triage the 4 sub-IQ4 IQ tiers x 2 models. Gated on the LCB retry freeing GPU1.
# DEGENERATE (>=2/3 bad prompts) -> touch <cell>.done (fence out of sweep).
# COHERENT -> rmdir <cell>.lock (re-open for the sweep's full HE+/MPE eval).
set +e
GPU=1; PORT=8242
GGUF_DIR=/mnt/sdc/ml/quant_sweep_gguf
WORK=/srv/ml/scripts/quant_sweep_work
LLAMA=/opt/llama.cpp/build/bin/llama-server
VERDICT=/srv/ml/logs/smoke_subiq4_verdict.txt
export HF_HUB_ENABLE_HF_TRANSFER=1
: > "$VERDICT"

echo "[smoke] waiting for lcb_retry_driver to finish (GPU1 free)..."
while pgrep -f lcb_retry_driver.sh >/dev/null 2>&1; do sleep 30; done
echo "[smoke] GPU1 free; start $(date -u)"

MODELS=(
  "v7coder|ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF|gemma-4-A4B-98e-v7-coder-it-|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it"
  "v7coderx|ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it-GGUF|gemma-4-A4B-98e-v7-coderx-it-|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it"
)
TIERS=(IQ2_M IQ2_XS IQ3_M IQ3_XXS)
PROMPTS=(
  "What is the capital of France? Answer with one word."
  "Write a Python function add(a, b) that returns a+b. Only the code."
  "Continue the sequence 2, 4, 6, 8 and give only the next number."
)

probe() {
  local port="$1" bad=0 i=0 p resp cls
  for p in "${PROMPTS[@]}"; do
    i=$((i+1))
    body=$(python3 -c 'import json,sys;print(json.dumps({"model":"smoke","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":256,"temperature":0}))' "$p")
    resp=$(curl -s -m 120 http://localhost:$port/v1/chat/completions -H 'Content-Type: application/json' -d "$body")
    cls=$(python3 - "$resp" <<'PY'
import json,sys,re
try:
    d=json.loads(sys.argv[1]); ch=d["choices"][0]; m=ch["message"]
    c=(m.get("content") or "") or (m.get("reasoning_content") or ""); fr=ch.get("finish_reason")
except Exception:
    print("BAD:parse"); sys.exit()
bad=(not c.strip()) or bool(re.search(r"(.)\1{29,}",c)) or (c.count("-")>=40) or (fr=="length" and len(c)>1500)
print(("BAD" if bad else "OK")+":"+repr(c[:70]))
PY
)
    echo "    q$i: $cls"
    [[ "$cls" == BAD* ]] && bad=$((bad+1))
  done
  return $bad
}

for spec in "${MODELS[@]}"; do
  IFS='|' read -r base repo prefix tok <<<"$spec"
  for t in "${TIERS[@]}"; do
    cell="${base}__${t}"; served="${base}-${t}"
    fname="${prefix}${t}.gguf"; gguf="$GGUF_DIR/$fname"
    echo "==== SMOKE $served $(date -u) ===="
    if [ ! -f "$gguf" ]; then
      echo "  [dl] $repo :: $fname"
      hf download "$repo" "$fname" --local-dir "$GGUF_DIR" >/dev/null 2>&1 || { echo "  [ERR] dl failed"; echo "$served DL_FAIL" >>"$VERDICT"; continue; }
    fi
    CUDA_VISIBLE_DEVICES=$GPU "$LLAMA" -m "$gguf" --port $PORT -c 8192 -ngl 99 --parallel 1 --no-warmup --jinja --reasoning off >/tmp/smoke_srv_$PORT.log 2>&1 &
    srv=$!
    for i in $(seq 1 90); do curl -s -m 3 http://localhost:$PORT/health 2>/dev/null | grep -q '"ok"\|"status":"ok"' && break; sleep 2; done
    probe $PORT; nbad=$?
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 2
    if [ "$nbad" -ge 2 ]; then
      verdict=DEGENERATE; rmdir "$WORK/$cell.lock" 2>/dev/null; : > "$WORK/$cell.done"
    else
      verdict=COHERENT; rmdir "$WORK/$cell.lock" 2>/dev/null
    fi
    echo "  => $served : $verdict (bad=$nbad/3)"
    echo "$served $verdict bad=$nbad/3" >>"$VERDICT"
    rm -f "$gguf"
  done
done
echo "[smoke] DONE $(date -u)"
echo "===== VERDICT ====="; cat "$VERDICT"
