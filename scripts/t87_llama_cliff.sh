#!/usr/bin/env bash
# T87 cliff-mapping — locate where the YaRN-extended ckpt's quality diverges from
# base as context grows past the 32k training length. VT (variable tracking, the
# axis that needs trained long-range attention) at 64k + 128k, ext vs base, on the
# SAME llama.cpp F16 setup as the 256k anchor. Reuses ruler_native_vt_256k via
# --metadata ctx_tokens override (no new template files; keeps max_tokens=120).
#
# Reference points already on record (llama.cpp F16, single-slot):
#   32k :  ext 0.92  base 0.948   |d|=0.028   (PASS — YaRN inert this deep)
#   256k:  ext 0.62  base 0.972   |d|=0.352   (FAIL — tracking untrained)
# Expectation if it's a 32k training-length wall: ext≈base at 64k, cliffs by 128k.
set -uo pipefail
LS=/opt/llama.cpp/build/bin/llama-server
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
OMK_PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK_BIN=$(dirname "$OMK_PY")
TOK=/srv/ml/google/gemma-4-26B-A4B-it
EXT=/srv/ml/longctx/t87_llama/ext-f16.gguf
BASE=/srv/ml/longctx/t87_llama/base-f16.gguf
RES=/srv/ml/longctx/ruler_llama
CTX=143360                                   # covers the 128k prompt + ~15k gen/headroom
YARN=(--rope-scaling yarn --yarn-orig-ctx 262144 --rope-scale 2.0 --yarn-beta-fast 32 --yarn-beta-slow 1)
KV=(--cache-type-k q8_0 --cache-type-v q8_0)

serve() {  # gguf port gpu yarn(0/1) alias logf -> pid
  local g="$1" p="$2" dev="$3" y="$4" al="$5" lf="$6"
  local ex=(); [ "$y" = 1 ] && ex=("${YARN[@]}")
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$LS" -m "$g" --port "$p" --host 127.0.0.1 \
     -ngl 99 -fa on --parallel 1 -c "$CTX" "${KV[@]}" "${ex[@]}" --alias "$al" --no-warmup \
     > "$RES/$lf" 2>&1 < /dev/null &
  echo $!
}
health() { for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
down() { pkill -f "llama-server.*--port $1" 2>/dev/null; sleep 2; }
tier() {  # gguf port served ctx label
  PATH="$OMK_BIN:$PATH" "$OMK_PY" "$OMK" --backend llama --no-server --port "$2" --parallel 1 \
     --model "$1" --served-name "$3" --tokenizer "$TOK" --template ruler_native_vt_256k \
     --metadata "ctx_tokens=$4" --results-dir "$RES/cliff_$5"
}
score() {  # served label
  "$OMK_PY" -c "import json;print(json.load(open('$RES/cliff_$2/ruler_native_vt_256k/$1/summary.json')).get('score'))" 2>/dev/null
}

for f in "$EXT" "$BASE"; do [ -f "$f" ] || { echo "FATAL: $f missing"; exit 2; }; done
echo "[cliff $(date '+%T %Z')] ext@${CTX}+YaRN GPU0:8201  ∥  base@${CTX} GPU1:8202"
serve "$EXT"  8201 0 1 ext  cliff_ext_serve.log  >/dev/null
serve "$BASE" 8202 1 0 base cliff_base_serve.log >/dev/null
health 8201 || { echo "FATAL: ext serve";  tail -25 "$RES/cliff_ext_serve.log";  down 8201; down 8202; exit 1; }
health 8202 || { echo "FATAL: base serve"; tail -25 "$RES/cliff_base_serve.log"; down 8201; down 8202; exit 1; }

for c in 65536 131072; do
  lab="$((c/1024))k"
  echo "[cliff $(date '+%T %Z')] VT @ ${lab} (ctx_tokens=$c)"
  ( tier "$EXT"  8201 ext  "$c" "$lab" ) & PE=$!
  ( tier "$BASE" 8202 base "$c" "$lab" ) & PB=$!
  wait $PE $PB
done
down 8201; down 8202

echo "=== CLIFF (llama.cpp F16 VT, ext=proportional⊕YaRN vs base=proportional) ==="
printf "%-7s %-8s %-8s %-8s\n" ctx ext base "|d|"
echo  "32k     0.92     0.948    0.028   (ref)"
for c in 65536 131072; do
  lab="$((c/1024))k"; e=$(score ext "$lab"); b=$(score base "$lab")
  d=$("$OMK_PY" -c "print(round(abs(($e)-($b)),3))" 2>/dev/null || echo NA)
  printf "%-7s %-8s %-8s %-8s\n" "$lab" "$e" "$b" "$d"
done
echo  "256k    0.62     0.972    0.352   (ref)"
echo "=== CLIFF DONE — read where |d| crosses ~0.10 to locate the training-length wall ==="
