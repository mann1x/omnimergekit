#!/usr/bin/env bash
# reprobe_iq2m.sh — gate the IQ2_M deletion. Fresh download + sha256-vs-sidecar
# integrity check + 6-prompt coherence probe, for both models' IQ2_M.
# Decides: published bytes genuinely degenerate (delete) vs corrupt-download fluke (hold).
set -uo pipefail
GPU=1; PORT=8246
DIR=/mnt/sdc/ml/reprobe_iq2m
WORK=/srv/ml/scripts/quant_sweep_work
LLAMA=/opt/llama.cpp/build/bin/llama-server
PY=/srv/ml/envs/envs/omnimergekit/bin/python
HF=/srv/ml/envs/envs/omnimergekit/bin/hf
OUT=/srv/ml/logs/reprobe_iq2m_verdict.txt
mkdir -p "$DIR"; : > "$OUT"

MODELS=(
  "v7coder|ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF|gemma-4-A4B-98e-v7-coder-it-IQ2_M.gguf"
  "v7coderx|ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it-GGUF|gemma-4-A4B-98e-v7-coderx-it-IQ2_M.gguf"
)
PROMPTS=(
  "What is the capital of France? Answer with one word."
  "Write a Python function add(a, b) that returns a+b. Only the code."
  "Continue the sequence 2, 4, 6, 8 and give only the next number."
  "What is 7 multiplied by 6? Give only the number."
  "Name three primary colors, comma-separated."
  "Translate 'good morning' into Spanish. Only the translation."
)

for spec in "${MODELS[@]}"; do
  IFS='|' read -r base repo fname <<<"$spec"
  echo "==== REPROBE $base IQ2_M  $(date -u) ===="
  rm -f "$DIR/$fname" "$DIR/$fname.sha256"
  "$HF" download "$repo" "$fname" --local-dir "$DIR" >/dev/null 2>"$DIR/.err" || { echo "  [ERR] gguf dl: $(tail -1 "$DIR/.err")"; echo "$base DL_FAIL" >>"$OUT"; continue; }
  "$HF" download "$repo" "$fname.sha256" --local-dir "$DIR" >/dev/null 2>/dev/null || true
  gguf="$DIR/$fname"
  # integrity: local sha256 vs published sidecar
  if [ -f "$DIR/$fname.sha256" ]; then
    pub=$(awk '{print $1}' "$DIR/$fname.sha256")
    loc=$(sha256sum "$gguf" | awk '{print $1}')
    [ "$pub" = "$loc" ] && integ="SHA_OK" || integ="SHA_MISMATCH(pub=$pub loc=$loc)"
  else
    integ="NO_SIDECAR"
  fi
  sz=$(stat -c%s "$gguf")
  echo "  bytes=$sz  integrity=$integ"
  CUDA_VISIBLE_DEVICES=$GPU "$LLAMA" -m "$gguf" --port $PORT -c 8192 -ngl 99 --parallel 1 --no-warmup --jinja --reasoning off >/tmp/reprobe_iq2m_$PORT.log 2>&1 &
  srv=$!
  for i in $(seq 1 90); do curl -s -m 3 http://localhost:$PORT/health 2>/dev/null | grep -q '"ok"\|"status":"ok"' && break; sleep 2; done
  bad=0; n=0
  for p in "${PROMPTS[@]}"; do
    n=$((n+1))
    body=$("$PY" -c 'import json,sys;print(json.dumps({"model":"r","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":256,"temperature":0}))' "$p")
    resp=$(curl -s -m 120 http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$body")
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
    echo "    q$n: $cls"
    [[ "$cls" == BAD* ]] && bad=$((bad+1))
  done
  kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 2
  [ "$bad" -ge 3 ] && v=DEGENERATE || v=COHERENT
  echo "  => $base IQ2_M : $v (bad=$bad/6) $integ"
  echo "$base IQ2_M $v bad=$bad/6 $integ" >>"$OUT"
  rm -f "$gguf" "$DIR/$fname.sha256"
done
echo "[reprobe] DONE $(date -u)"
echo "===== VERDICT ====="; cat "$OUT"
