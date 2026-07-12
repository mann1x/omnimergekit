#!/usr/bin/env bash
# build_v7_qat.sh — build QAT-Q4_0 GGUF(s) of a v7 model by re-running the
# published v7 prune surgery on Google's QAT-unquantized base.
#
# A "-qat" v7 = the SAME prune recipe (expert_drop with the SAME drop map, then
# the published shared-α=1.2 step) applied to
#   google/gemma-4-26B-A4B-it-qat-q4_0-unquantized   (q4_0-aware-trained bf16)
# then cast to Q4_0 (no imatrix — QAT calibration is baked in).
#
# Mirrors /srv/ml/scripts/build_v7coder.sh, source dir swapped to the QAT base
# and the drop map REUSED verbatim (the drop is a function of competence data,
# not base weights; the QAT base is "the same model, different checkpoint").
#
# Usage:  build_v7_qat.sh <g15f2440|fs2440> <both|shared|noshared>
#   both     -> build noshared Q4_0 first (pristine drop), then shared α1.2 Q4_0
#   shared   -> only the shared α1.2 Q4_0
#   noshared -> only the drop-only Q4_0
set -uo pipefail

MODEL="${1:?usage: build_v7_qat.sh <g15f2440|fs2440> <both|shared|noshared>}"
VARIANT="${2:?usage: build_v7_qat.sh <g15f2440|fs2440> <both|shared|noshared>}"

BM=/srv/ml
PY="$BM/envs/envs/omnimergekit/bin/python"
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
SCR="$BM/scripts"
CONVERT="$BM/tools/llama.cpp/convert_hf_to_gguf.py"
QUANT=/opt/llama.cpp/build/bin/llama-quantize
LLAMA_CLI=/opt/llama.cpp/build/bin/llama-cli
QAT_BASE=/mnt/sdc/ml/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
QAT_REPO=google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
OUTGG=/mnt/sdc/ml/eval_gguf/qat
WORKROOT=/mnt/sdc/ml/google/qat_build
mkdir -p "$OUTGG" "$WORKROOT"

case "$MODEL" in
  g15f2440) PUB_DIR=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it
            MAP="$SCR/v7coder_g15f2440_drop_map.json"
            CLEAN=gemma-4-A4B-98e-v7-coder-it ;;
  fs2440)   PUB_DIR=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it
            MAP="$SCR/v7coder_fs2440_drop_map.json"
            CLEAN=gemma-4-A4B-98e-v7-coderx-it ;;
  *) echo "[FATAL] unknown model '$MODEL' (want g15f2440|fs2440)"; exit 1 ;;
esac
case "$VARIANT" in both|shared|noshared) :;; *) echo "[FATAL] bad variant '$VARIANT'"; exit 1;; esac

DIR="$WORKROOT/$CLEAN-qat"                 # expert_drop output = pristine drop (noshared)
GG_NOSHARED="$OUTGG/$CLEAN-qat-noshared-Q4_0.gguf"
GG_SHARED="$OUTGG/$CLEAN-qat-Q4_0.gguf"
LOG="$WORKROOT/build_${MODEL}_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
L(){ echo "[qat $(date -u +%H:%M:%S)] $*"; }

L "=== build_v7_qat  model=$MODEL variant=$VARIANT clean=$CLEAN ==="

# ---- preflight ----
for f in "$PY" "$SCR/expert_drop.py" "$SCR/router_shared_upweight.py" "$CONVERT" "$QUANT" "$MAP" "$PUB_DIR"; do
  [ -e "$f" ] || { L "FATAL missing: $f"; exit 1; }
done
DPL=$("$PY" -c "import json,statistics as s;d=json.load(open('$MAP'));v=[len(x) for x in d.values() if isinstance(x,list)];print(int(s.mean(v)) if v else -1)")
L "drop map $MAP -> dropped/layer=$DPL (expect 30 -> 98e)"
[ "$DPL" = "30" ] || { L "FATAL drop map not 98e (dropped/layer=$DPL)"; exit 1; }

# ---- QAT base (download once) ----
if [ ! -f "$QAT_BASE/model.safetensors.index.json" ]; then
  L "downloading QAT base $QAT_REPO -> $QAT_BASE"
  hf download "$QAT_REPO" --local-dir "$QAT_BASE" >/dev/null 2>"$WORKROOT/qat_dl.err" \
    || { L "FATAL QAT base download failed"; sed -n '1,8p' "$WORKROOT/qat_dl.err"; exit 1; }
fi
[ -f "$QAT_BASE/model.safetensors.index.json" ] || { L "FATAL QAT base incomplete"; exit 1; }

# ---- [1] expert_drop (pristine = noshared) ----
if [ -f "$DIR/model.safetensors.index.json" ]; then
  L "[1] expert_drop output exists, skip ($DIR)"
else
  L "[1] expert_drop QAT_base + $MODEL map -> $DIR"
  "$PY" "$SCR/expert_drop.py" --source-dir "$QAT_BASE" --drop-map "$MAP" --output-dir "$DIR" 2>&1 | tail -8
  [ -f "$DIR/model.safetensors.index.json" ] || { L "FATAL expert_drop failed"; exit 1; }
fi
# ensure aux configs convert needs (copy from the published v7 bf16 dir if absent)
for aux in preprocessor_config.json processor_config.json generation_config.json tokenizer.json tokenizer_config.json; do
  if [ ! -f "$DIR/$aux" ] && [ -f "$PUB_DIR/$aux" ]; then cp -n "$PUB_DIR/$aux" "$DIR/$aux"; L "  copied aux $aux from published dir"; fi
done

# ---- helpers ----
gguf_ok(){ # magic-header check: first 4 bytes must be 'GGUF'
  local f="$1"; [ -s "$f" ] || return 1
  local magic; magic=$("$PY" -c "import sys;print(open('$f','rb').read(4).decode('latin1'))" 2>/dev/null)
  [ "$magic" = "GGUF" ]
}
sanity(){ # quick CPU coherence check (non-fatal)
  local gg="$1"
  [ -x "$LLAMA_CLI" ] || { L "  (sanity skip: no llama-cli)"; return 0; }
  local out
  out=$(CUDA_VISIBLE_DEVICES="" "$LLAMA_CLI" -m "$gg" -ngl 0 --no-warmup -no-cnv -n 24 -p "The capital of France is" 2>/dev/null | tr -d '\n' | tail -c 200)
  L "  sanity('capital of France is'): ...${out: -120}"
  printf '%s' "$out" | grep -qi "Paris" && L "  sanity OK (Paris)" || L "  sanity WARN (Paris not seen — inspect)"
}
quant_q40(){ # $1 src_dir  $2 out_gguf
  local src="$1" outgg="$2"
  if gguf_ok "$outgg"; then L "  Q4_0 exists+valid, skip: $(basename "$outgg")"; return 0; fi
  local f16="$OUTGG/.$(basename "$outgg" .gguf).F16.gguf"
  L "  convert -> F16"
  "$PY" "$CONVERT" "$src" --outfile "$f16" --outtype f16 2>&1 | tail -3
  local n; n=$("$PY" -c "from gguf.gguf_reader import GGUFReader;print(len(GGUFReader('$f16').tensors))" 2>/dev/null)
  L "  F16 tensors=$n"; [ "${n:-0}" -lt 600 ] && { L "FATAL too few tensors ($n)"; rm -f "$f16"; exit 1; }
  L "  llama-quantize Q4_0 (no imatrix) -> $(basename "$outgg")"
  "$QUANT" "$f16" "$outgg" Q4_0 2>&1 | tail -3
  rm -f "$f16"
  gguf_ok "$outgg" || { L "FATAL Q4_0 invalid header: $outgg"; exit 1; }
  ls -la "$outgg"; sanity "$outgg"
}

# ---- [2] noshared Q4_0 (must precede the in-place shared step) ----
if [ "$VARIANT" = both ] || [ "$VARIANT" = noshared ]; then
  L "[2] noshared Q4_0"
  quant_q40 "$DIR" "$GG_NOSHARED"
fi

# ---- [3] shared α1.2 Q4_0 ----
if [ "$VARIANT" = both ] || [ "$VARIANT" = shared ]; then
  if [ ! -f "$DIR/.shared_applied" ]; then
    L "[3] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight (in place on $DIR)"
    "$PY" "$SCR/router_shared_upweight.py" --model-dir "$DIR" --alpha 1.2 --target mlp.down_proj.weight 2>&1 | tail -6
    rm -f "$DIR"/*.pre_shared_upweight
    touch "$DIR/.shared_applied"
  else
    L "[3] .shared_applied exists, skip upweight"
  fi
  L "[3] shared Q4_0"
  quant_q40 "$DIR" "$GG_SHARED"
fi

# ---- [4] cleanup bf16 dir (GGUFs kept) ----
L "[4] removing bf16 work dir $DIR (GGUFs retained in $OUTGG)"
rm -rf "$DIR"

L "=== DONE: built ->"
[ -f "$GG_NOSHARED" ] && ls -la "$GG_NOSHARED"
[ -f "$GG_SHARED" ]   && ls -la "$GG_SHARED"
L "###### BUILD_V7_QAT_DONE $MODEL $VARIANT ######"
