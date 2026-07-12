#!/usr/bin/env bash
# T87 llama.cpp RULER anchor — serve F16 GGUF with YaRN applied at RUNTIME (the
# proportional base rope is baked into the GGUF; --rope-scaling yarn composes the
# extension = proportional_yarn). Compares ext@256k vs base@256k on the SAME backend.
#
#   t87_llama_anchor.sh smoke    # serve ext+yarn @32k, run vt_32k (fast gate)
#   t87_llama_anchor.sh anchor   # ext@256k+yarn (GPU0) ∥ base@256k (GPU1), vt+mk1
set -uo pipefail
PHASE="${1:-smoke}"
LS=/opt/llama.cpp/build/bin/llama-server
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
OMK_PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK_BIN=$(dirname "$OMK_PY")
TOK=/srv/ml/google/gemma-4-26B-A4B-it                 # RULER prepare tokenizer (clean base)
EXT=/srv/ml/longctx/t87_llama/ext-f16.gguf
BASE=/srv/ml/longctx/t87_llama/base-f16.gguf
RES=/srv/ml/longctx/ruler_llama
mkdir -p "$RES"
# YaRN runtime rope: proportional base (in GGUF) + YaRN ramp. orig_ctx=256k, factor 2 → 512k.
YARN=(--rope-scaling yarn --yarn-orig-ctx 262144 --rope-scale 2.0 --yarn-beta-fast 32 --yarn-beta-slow 1)
KV=(--cache-type-k q8_0 --cache-type-v q8_0)

serve() {  # gguf port gpu ctx yarn(0/1) alias logf  -> echoes pid
  local g="$1" p="$2" dev="$3" ctx="$4" y="$5" al="$6" lf="$7"
  local ex=(); [ "$y" = 1 ] && ex=("${YARN[@]}")
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$LS" -m "$g" --port "$p" --host 127.0.0.1 \
     -ngl 99 -fa on --parallel 1 -c "$ctx" "${KV[@]}" "${ex[@]}" --alias "$al" --no-warmup \
     > "$RES/$lf" 2>&1 < /dev/null &
  echo $!
}
health() {  # port  (30 min budget — 256k F16 load is slow)
  local p="$1"
  for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}
down() { pkill -f "llama-server.*--port $1" 2>/dev/null; sleep 2; }
tier() {  # gguf port served template
  PATH="$OMK_BIN:$PATH" "$OMK_PY" "$OMK" --backend llama --no-server --port "$2" --parallel 1 \
     --model "$1" --served-name "$3" --tokenizer "$TOK" --template "$4" --results-dir "$RES"
}
score() {  # template served
  "$OMK_PY" -c "import json;print(json.load(open('$RES/$1/$2/summary.json')).get('score'))" 2>/dev/null
}

if [ "$PHASE" = smoke ]; then
  [ -f "$EXT" ] || { echo "FATAL: $EXT missing"; exit 2; }
  echo "[smoke $(date '+%T %Z')] serve ext F16 + YaRN @ ctx=40960 GPU0:8201"
  serve "$EXT" 8201 0 40960 1 ext smoke_serve.log >/dev/null
  health 8201 || { echo "FATAL: ext serve unhealthy"; tail -25 "$RES/smoke_serve.log"; down 8201; exit 1; }
  echo "[smoke] run vt_32k"
  tier "$EXT" 8201 ext ruler_native_vt_32k
  down 8201
  S=$(score ruler_native_vt_32k ext)
  echo "=== SMOKE vt_32k (ext F16 + YaRN, llama.cpp) = ${S}   (base ref 0.948) ==="
  if awk "BEGIN{exit !(\"$S\"+0>=0.80)}" 2>/dev/null; then
    echo "[smoke] PASS — proportional⊕YaRN faithful at native length; run: t87_llama_anchor.sh anchor"
  else
    echo "[smoke] FAIL — yarn flags or rope still off (score $S)"
  fi
  exit 0
fi

if [ "$PHASE" = anchor ]; then
  for f in "$EXT" "$BASE"; do [ -f "$f" ] || { echo "FATAL: $f missing"; exit 2; }; done
  echo "[anchor $(date '+%T %Z')] ext@256k+YaRN GPU0:8201  ∥  base@256k GPU1:8202"
  serve "$EXT"  8201 0 270336 1 ext  anchor_ext_serve.log  >/dev/null   # 262144 prompt + headroom for gen
  serve "$BASE" 8202 1 270336 0 base anchor_base_serve.log >/dev/null
  health 8201 || { echo "FATAL: ext serve"; tail -25 "$RES/anchor_ext_serve.log"; down 8201; down 8202; exit 1; }
  health 8202 || { echo "FATAL: base serve"; tail -25 "$RES/anchor_base_serve.log"; down 8201; down 8202; exit 1; }
  ( tier "$EXT"  8201 ext  ruler_native_vt_256k; tier "$EXT"  8201 ext  ruler_native_mk1_256k ) & P1=$!
  ( tier "$BASE" 8202 base ruler_native_vt_256k; tier "$BASE" 8202 base ruler_native_mk1_256k ) & P2=$!
  wait $P1 $P2
  down 8201; down 8202
  echo "=== ANCHOR (llama.cpp F16, 256k, ext=proportional⊕YaRN vs base=proportional) ==="
  FAIL=0
  for t in ruler_native_vt_256k ruler_native_mk1_256k; do
    e=$(score "$t" ext); b=$(score "$t" base)
    d=$("$OMK_PY" -c "print(abs(($e)-($b)))" 2>/dev/null || echo NA)
    echo "  $t  ext=$e  base=$b  |d|=$d"
    awk "BEGIN{exit !(\"$d\"!=\"NA\" && \"$d\"+0<=0.10)}" 2>/dev/null || FAIL=1
  done
  [ "$FAIL" = 0 ] && echo "=== RESULT: PASS (tol=0.10) — llama.cpp serves the extension faithfully ===" \
                  || echo "=== RESULT: FAIL — even llama.cpp diverges; transformers tiebreaker next ==="
  exit 0
fi

echo "usage: $0 {smoke|anchor}"; exit 2
