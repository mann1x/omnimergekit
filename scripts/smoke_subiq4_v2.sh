#!/usr/bin/env bash
# smoke_subiq4_v2.sh — re-run the sub-IQ4 coherence triage after the v1 run
# produced false DL_FAILs (it swallowed hf-download errors to /dev/null while
# the disk was under contention). Harness validated on Q4_K_M (COHERENT) first.
#
# Fixes vs v1:
#   - download errors are VISIBLE + retried once (no /dev/null swallow)
#   - releases any stale .lock before probing
#   - DEGENERATE (>=2/3 bad) -> touch <cell>.done (fence out of sweep) + rmdir .lock
#   - COHERENT              -> rmdir .lock (re-open for the sweep's full eval)
#   - NO lcb_retry_driver gate (that driver is dead)
#
# HF removal of any DEGENERATE tier is a SEPARATE, user-authorized step — this
# script only fences the sweep; it never deletes anything from HF or ollama.
set -uo pipefail
GPU=1; PORT=8242
GGUF_DIR=/mnt/sdc/ml/quant_sweep_gguf
WORK=/srv/ml/scripts/quant_sweep_work
LLAMA=/opt/llama.cpp/build/bin/llama-server
PY=/srv/ml/envs/envs/omnimergekit/bin/python
HF=/srv/ml/envs/envs/omnimergekit/bin/hf
VERDICT=/srv/ml/logs/smoke_subiq4_v2_verdict.txt
mkdir -p "$WORK"; : > "$VERDICT"

MODELS=(
  "v7coder|ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF|gemma-4-A4B-98e-v7-coder-it-"
  "v7coderx|ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it-GGUF|gemma-4-A4B-98e-v7-coderx-it-"
)
TIERS=(IQ2_M IQ2_XS IQ3_M IQ3_XXS)
PROMPTS=(
  "What is the capital of France? Answer with one word."
  "Write a Python function add(a, b) that returns a+b. Only the code."
  "Continue the sequence 2, 4, 6, 8 and give only the next number."
)

probe() {
  local port="$1" bad=0 i=0 p resp cls body
  for p in "${PROMPTS[@]}"; do
    i=$((i+1))
    body=$("$PY" -c 'import json,sys;print(json.dumps({"model":"smoke","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":256,"temperature":0}))' "$p")
    resp=$(curl -s -m 120 http://localhost:$port/v1/chat/completions -H 'Content-Type: application/json' -d "$body")
    cls=$("$PY" - "$resp" <<'PYEOF'
import json,sys,re
try:
    d=json.loads(sys.argv[1]); ch=d["choices"][0]; m=ch["message"]
    c=(m.get("content") or "") or (m.get("reasoning_content") or ""); fr=ch.get("finish_reason")
except Exception as e:
    print("BAD:parse:%s"%e); sys.exit()
bad=(not c.strip()) or bool(re.search(r"(.)\1{29,}",c)) or (c.count("-")>=40) or (fr=="length" and len(c)>1500)
print(("BAD" if bad else "OK")+":"+repr(c[:70]))
PYEOF
)
    echo "    q$i: $cls"
    [[ "$cls" == BAD* ]] && bad=$((bad+1))
  done
  return $bad
}

dl() {  # repo fname  -> 0 ok / 1 fail (visible errors + 1 retry)
  local repo="$1" fname="$2" a
  for a in 1 2; do
    if "$HF" download "$repo" "$fname" --local-dir "$GGUF_DIR" 2>"$WORK/.dl_err"; then return 0; fi
    echo "  [dl attempt $a FAILED] $(tail -2 "$WORK/.dl_err" | tr '\n' ' ')"
    sleep 5
  done
  return 1
}

echo "[smoke-v2] START $(date -u)  gpu=$GPU port=$PORT"
for spec in "${MODELS[@]}"; do
  IFS='|' read -r base repo prefix <<<"$spec"
  for t in "${TIERS[@]}"; do
    cell="${base}__${t}"; served="${base}-${t}"
    fname="${prefix}${t}.gguf"; gguf="$GGUF_DIR/$fname"
    echo "==== SMOKE $served $(date -u) ===="
    if [ ! -f "$gguf" ]; then
      echo "  [dl] $repo :: $fname"
      if ! dl "$repo" "$fname"; then
        echo "  [ERR] download failed after retry"; echo "$served DL_FAIL" >>"$VERDICT"; continue
      fi
    fi
    CUDA_VISIBLE_DEVICES=$GPU "$LLAMA" -m "$gguf" --port $PORT -c 8192 -ngl 99 --parallel 1 --no-warmup --jinja --reasoning off >/tmp/smoke_v2_srv_$PORT.log 2>&1 &
    srv=$!
    for i in $(seq 1 90); do curl -s -m 3 http://localhost:$PORT/health 2>/dev/null | grep -q '"ok"\|"status":"ok"' && break; sleep 2; done
    probe $PORT; nbad=$?
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 2
    rmdir "$WORK/$cell.lock" 2>/dev/null  # release stale fence
    if [ "$nbad" -ge 2 ]; then
      verdict=DEGENERATE; : > "$WORK/$cell.done"     # fence out of the sweep
    else
      verdict=COHERENT                                 # left unfenced -> sweep will eval
    fi
    echo "  => $served : $verdict (bad=$nbad/3)"
    echo "$served $verdict bad=$nbad/3" >>"$VERDICT"
    rm -f "$gguf"
  done
done
echo "[smoke-v2] DONE $(date -u)"
echo "===== VERDICT ====="; cat "$VERDICT"
