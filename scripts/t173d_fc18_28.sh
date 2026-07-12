#!/bin/bash
# T173d — codelcb43 weights + RAISED floor 18-28, chasing 0% looping.
# Single variant, single GPU, fully sequential (no background wait -> no deadlock).
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
AUDIT=/srv/ml/scripts/audit_full_bench.py
LOOPCK=/srv/ml/repos/omnimergekit/scripts/loop_precision_check.py
SCR=/srv/ml/repos/omnimergekit/scripts
RES=/srv/ml/eval_results_tracks_2_3
BASE=/srv/ml/google/gemma-4-26B-A4B-it
A2_IMATRIX=/srv/ml/models/variants/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-GGUF/imatrix.dat
WORK=/mnt/sdc/ml/google
PRISTINE=gemma-4-A4B-62e-fc15_25-p8-pristine-it
NAME=gemma-4-A4B-62e-fc18_28-p8-codelcb43-it
MAP=$SCR/v6coder_C6v3lcb_62e_fc18_28_p8_codelcb43_drop_map.json
BENCH=ifeval_100
TLIM=5400
export PATH="/srv/ml/envs/envs/omnimergekit/bin:${PATH:-/usr/bin:/bin}"
export LD_LIBRARY_PATH="/srv/ml/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
CONVERT=/srv/ml/tools/llama.cpp/convert_hf_to_gguf.py
[ -f "$CONVERT" ] || CONVERT=/workspace/llama.cpp/convert_hf_to_gguf.py
[ -f "$CONVERT" ] || CONVERT=/opt/llama.cpp/convert_hf_to_gguf.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
[ -x "$QUANT" ] || QUANT=/workspace/llama.cpp/build/bin/llama-quantize
LOG_DIR=/srv/ml/logs/t173; mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
exec > >(tee "$LOG_DIR/t173d_fc18_28_${TS}.log") 2>&1

out="$WORK/$NAME"; gdir="$WORK/$NAME-GGUF"; q6="$gdir/${NAME}-Q6_K.gguf"; f16="$gdir/${NAME}-F16.gguf"
echo "==================== T173d BUILD $NAME ===================="
if [ -f "$q6" ]; then echo "[build] SKIP (Q6_K exists)"; else
  [ -f "$MAP" ] || { echo "FATAL map missing $MAP"; exit 1; }
  rm -rf "$out"
  "$PY" "$SCR/expert_drop.py" --source-dir "$BASE" --drop-map "$MAP" --output-dir "$out" || { echo "FATAL expert_drop"; exit 1; }
  [ -f "$out/model.safetensors.index.json" ] || { echo "FATAL no index"; exit 1; }
  for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
           preprocessor_config.json processor_config.json special_tokens_map.json; do
      [ -s "$out/$f" ] || { [ -f "$BASE/$f" ] && cp "$BASE/$f" "$out/$f"; }
  done
  mkdir -p "$gdir"
  echo "[build] convert -> F16"; "$PY" "$CONVERT" "$out" --outfile "$f16" --outtype f16 2>&1 | tail -4
  [ -f "$f16" ] || { echo "FATAL F16"; exit 1; }
  echo "[build] quantize -> Q6_K (A2 imatrix)"; "$QUANT" --imatrix "$A2_IMATRIX" "$f16" "$q6" Q6_K 2>&1 | tail -4
  [ -f "$q6" ] || { echo "FATAL Q6_K"; exit 1; }
  for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
           preprocessor_config.json processor_config.json special_tokens_map.json config.json; do
      [ -s "$out/$f" ] && [ ! -s "$gdir/$f" ] && cp "$out/$f" "$gdir/$f"
  done
  [ -e "$gdir/imatrix.dat" ] || ln "$A2_IMATRIX" "$gdir/imatrix.dat" 2>/dev/null || cp "$A2_IMATRIX" "$gdir/imatrix.dat"
  rm -f "$f16"; rm -rf "$out"
  echo "[build $(date +%H:%M:%S)] DONE q6=$(du -h "$q6"|cut -f1)"
fi

echo "==================== T173d EVAL ifeval_100 (GPU0) ===================="
sd="$RES/$BENCH/$NAME"; [ -d "$sd" ] && rm -rf "$sd"
pkill -KILL -f "llama-server.*--port 8295" 2>/dev/null; sleep 2
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=10 --signal=KILL "$TLIM" \
    "$PY" "$OMK" --model "$q6" --tokenizer "$gdir" --template "$BENCH" \
    --backend llama --port 8295 --served-name "$NAME" --results-dir "$RES" 2>&1 | tail -8
pkill -KILL -f "llama-server.*--port 8295" 2>/dev/null; sleep 2

echo "==================== T173d AUDIT vs pristine ===================="
"$PY" "$AUDIT" "$BENCH" "$PRISTINE" "$PRISTINE" 2>/dev/null | grep "^AUDIT" || true
"$PY" "$AUDIT" "$BENCH" "$NAME" "$PRISTINE" 2>/dev/null | grep "^AUDIT" || echo "AUDIT_FAIL $NAME"
"$PY" "$LOOPCK" "$BENCH" "$NAME" 2>/dev/null | grep -E "loop-flagged|TOTAL|doc=" || true
echo "==================== T173d DONE $(date +%H:%M:%S) ===================="
