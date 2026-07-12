#!/usr/bin/env bash
# Controlled v6 GGUF: rebuild v6 Q6_K ON bs2 from the downloaded bf16 (same convert+quant
# toolchain as v7/fs2440), eval HE+/LCB on bs2 llama-server. Isolates HF-GGUF-build vs
# bs2-inference as the cause of the HF-v6 over-generation (4 LCB truncations -> 48/55).
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${1:-0}"
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
SRC=/mnt/sdc/ml/google/gemma-4-A4B-98e-v6-coder-it
GG=$SRC-bs2built-GGUF
F16=$GG/v6coder-bs2built-F16.gguf
Q6=$GG/v6coder-bs2built-Q6_K.gguf
CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
NAME=v6coder-bs2built-q6k
RES=$BM/eval_results_v6coder_bs2built
PORT="${2:-8195}"
L(){ echo "[v6rebuild $(date -u +%H:%M:%S)] $*"; }
mkdir -p "$GG" "$RES"
[ -f "$SRC/config.json" ] || { L "FATAL: v6 bf16 missing at $SRC"; exit 1; }
if [ ! -f "$Q6" ]; then
  L "convert bf16 -> F16"; "$PY" "$CONVERT" "$SRC" --outfile "$F16" --outtype f16 2>&1 | tail -3
  n=$("$PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader(\"$F16\").tensors))")
  L "F16 tensors=$n"; [ "$n" -lt 600 ] && { L "FATAL tensors=$n"; exit 1; }
  L "quant Q6_K"; "$QUANT" "$F16" "$Q6" Q6_K 2>&1 | tail -3; rm -f "$F16"
fi
ls -la "$Q6"
sc(){ "$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get(\"score\"))" "$1" 2>/dev/null; }
for tpl in humanevalplus_full lcb_medium_55_v4; do
  L ">>> eval $tpl (GPU$CUDA_VISIBLE_DEVICES port $PORT)"
  "$PY" "$OMK" --model "$Q6" --tokenizer "$SRC" --template "$tpl" --backend llama --port "$PORT" --served-name "$NAME" --results-dir "$RES" 2>&1 | tail -15
  L "SCORE $tpl = $(sc "$RES/$tpl/$NAME/summary.json")"
done
L "V6REBUILD_DONE he+=$(sc "$RES/humanevalplus_full/$NAME/summary.json") lcb=$(sc "$RES/lcb_medium_55_v4/$NAME/summary.json")"
