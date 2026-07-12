#!/bin/bash
# T173c — code/lcb-ward 62e variants: codelcb43 (weight bump) + fc12_22 (floor lower).
# Loop-test vs gemma-4-A4B-62e-fc15_25-p8-pristine-it. Apples-to-apples (no shared/PES).
# FIX vs t173_build_eval.sh: explicit-pid `wait` (bare wait deadlocked on the exec>tee coproc).
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
exec > >(tee "$LOG_DIR/t173c_build_eval_${TS}.log") 2>&1

VARIANTS=(
"gemma-4-A4B-62e-fc15_25-p8-codelcb43-it|$SCR/v6coder_C6v3lcb_62e_fc15_25_p8_codelcb43_drop_map.json"
"gemma-4-A4B-62e-fc12_22-p8-it|$SCR/v6coder_C6v3lcb_62e_fc12_22_p8_drop_map.json"
)

build_one() {
    local name=$1 map=$2
    local out="$WORK/$name" gdir="$WORK/$name-GGUF"
    local q6="$gdir/${name}-Q6_K.gguf" f16="$gdir/${name}-F16.gguf"
    if [ -f "$q6" ]; then echo "[build] SKIP $name (Q6_K exists)"; return 0; fi
    echo "[build $(date +%H:%M:%S)] $name  map=$(basename "$map")"
    [ -f "$map" ] || { echo "FATAL map missing $map"; return 1; }
    rm -rf "$out"
    "$PY" "$SCR/expert_drop.py" --source-dir "$BASE" --drop-map "$map" --output-dir "$out" || { echo "FATAL expert_drop $name"; return 1; }
    [ -f "$out/model.safetensors.index.json" ] || { echo "FATAL no index $name"; return 1; }
    for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
             preprocessor_config.json processor_config.json special_tokens_map.json; do
        [ -s "$out/$f" ] || { [ -f "$BASE/$f" ] && cp "$BASE/$f" "$out/$f"; }
    done
    mkdir -p "$gdir"
    echo "[build] convert -> F16"
    "$PY" "$CONVERT" "$out" --outfile "$f16" --outtype f16 2>&1 | tail -4
    [ -f "$f16" ] || { echo "FATAL F16 $name"; return 1; }
    echo "[build] quantize -> Q6_K (A2 imatrix)"
    "$QUANT" --imatrix "$A2_IMATRIX" "$f16" "$q6" Q6_K 2>&1 | tail -4
    [ -f "$q6" ] || { echo "FATAL Q6_K $name"; return 1; }
    for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
             preprocessor_config.json processor_config.json special_tokens_map.json config.json; do
        [ -s "$out/$f" ] && [ ! -s "$gdir/$f" ] && cp "$out/$f" "$gdir/$f"
    done
    [ -e "$gdir/imatrix.dat" ] || ln "$A2_IMATRIX" "$gdir/imatrix.dat" 2>/dev/null || cp "$A2_IMATRIX" "$gdir/imatrix.dat"
    rm -f "$f16"; rm -rf "$out"
    echo "[build $(date +%H:%M:%S)] DONE $name  q6=$(du -h "$q6"|cut -f1)"
}

eval_one() {
    local name=$1 gpu=$2 port=$3
    local gdir="$WORK/$name-GGUF" q6="$WORK/$name-GGUF/${name}-Q6_K.gguf"
    local sd="$RES/$BENCH/$name"
    [ -d "$sd" ] && rm -rf "$sd"
    pkill -KILL -f "llama-server.*--port $port" 2>/dev/null; sleep 2
    echo "[eval gpu$gpu $(date +%H:%M:%S)] $name"
    CUDA_VISIBLE_DEVICES=$gpu timeout --kill-after=10 --signal=KILL "$TLIM" \
        "$PY" "$OMK" --model "$q6" --tokenizer "$gdir" --template "$BENCH" \
        --backend llama --port "$port" --served-name "$name" --results-dir "$RES" 2>&1 \
        | sed "s/^/[gpu$gpu] /" | tail -6
    pkill -KILL -f "llama-server.*--port $port" 2>/dev/null; sleep 2
    echo "[eval gpu$gpu $(date +%H:%M:%S)] DONE $name"
}

echo "==================== T173c BUILD (sequential) ===================="
for v in "${VARIANTS[@]}"; do build_one "${v%%|*}" "${v#*|}" || exit 1; done

echo "==================== wait for logcreat20 recovery to free GPU0 ===================="
while pgrep -f "omk_eval.*logcreat20" >/dev/null 2>&1; do echo "[guard $(date +%H:%M:%S)] logcreat20 still evaling, wait"; sleep 30; done
echo "[guard $(date +%H:%M:%S)] GPU0 free"

echo "==================== T173c EVAL (dual GPU) ===================="
NA="${VARIANTS[0]%%|*}"; NB="${VARIANTS[1]%%|*}"
eval_one "$NA" 0 8295 & pA=$!
eval_one "$NB" 1 8296 & pB=$!
wait "$pA" "$pB"          # explicit pids — does NOT block on the tee coproc

echo "==================== T173c AUDIT vs pristine ===================="
"$PY" "$AUDIT" "$BENCH" "$PRISTINE" "$PRISTINE" 2>/dev/null | grep "^AUDIT" || true
for v in "${VARIANTS[@]}"; do
    nm="${v%%|*}"
    "$PY" "$AUDIT" "$BENCH" "$nm" "$PRISTINE" 2>/dev/null | grep "^AUDIT" || echo "AUDIT_FAIL $nm"
done
for v in "${VARIANTS[@]}"; do "$PY" "$LOOPCK" "$BENCH" "${v%%|*}" 2>/dev/null | grep -E "loop-flagged|TOTAL"; done
echo "==================== T173c DONE $(date +%H:%M:%S) ===================="
