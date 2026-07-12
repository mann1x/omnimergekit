#!/usr/bin/env bash
# build_std16_qat_tiers.sh — STD16 (v7-coder force-keep promotion candidate) QAT tiers.
#   qat-Q4_0      = expert_drop(google QAT ckpt, v8coder_fk16 map) -> shared a1.2 -> F16 -> Q4_0
#                   (NO imatrix; QAT calibration baked in; published winner = shared variant)
#   CD-qat-Q4_K_M = same STD16 QAT F16 + qat imatrix (calib_both) + CD-Q4_K_M tensor-type-file
# Recipe byte-faithful to build_qat_pruned_base.sh + build_v7_qat.sh, drop map swapped to fk16.
# CPU work (base F16 + qat-Q4_0) runs immediately; the GPU imatrix waits until the plain loop
# gate's done-markers are all present (no GPU contention), then CD-qat is cut. Outputs -> cohort
# GGUF dir. The qat imatrix is archived next to the quants (MANDATORY preservation rule).
set -uo pipefail
BM=/srv/ml
PY="$BM/envs/envs/omnimergekit/bin/python"
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
SCR="$BM/scripts"
CONVERT="$BM/tools/llama.cpp/convert_hf_to_gguf.py"
BIN=/opt/llama.cpp/build/bin
QAT_BASE=/mnt/sdc/ml/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
MAP="$SCR/v8coder_fk16_drop_map.json"                         # STD16 drop map
AUX=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it      # tokenizer/preprocessor source
CD="$SCR/cd_maps_v7_fixed/coder/tensor_types_CD-Q4_K_M.txt"
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt
WORK=/mnt/sdc/ml/t223_fk
DIR="$WORK/std16-qat-pruned"
F16="$WORK/std16-qat-F16.gguf"
IMAT="$WORK/std16-qat-imatrix.dat"
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
Q40="$GG/gemma-4-A4B-98e-v7-coder-it-qat-Q4_0.gguf"
CDQAT="$GG/gemma-4-A4B-98e-v7-coder-it-CD-qat-Q4_K_M.gguf"
DONE=/mnt/sdc/ml/std16_gate/done
PLAIN_NEED="Q4_K_S Q4_K_M Q4_K_L Q5_K_M Q5_K_L Q6_K_L Q8_0"
LOG="$WORK/build_std16_qat_tiers.log"
exec > >(tee -a "$LOG") 2>&1
L(){ echo "[std16qat $(date -u +%T)] $*"; }
magic(){ "$PY" -c "print(open('$1','rb').read(4).decode('latin1'))" 2>/dev/null; }
gguf_ok(){ [ -s "$1" ] && [ "$(magic "$1")" = GGUF ]; }

L "=== STD16 QAT tiers (fk16 map) ==="
for f in "$PY" "$SCR/expert_drop.py" "$SCR/router_shared_upweight.py" "$CONVERT" "$MAP" \
         "$QAT_BASE" "$AUX" "$CD" "$CALIB" "$BIN/llama-quantize" "$BIN/llama-imatrix"; do
  [ -e "$f" ] || { L "FATAL missing $f"; exit 1; }
done
[ "$(grep -cE '=IQ' "$CD")" -eq 0 ] || { L "FATAL CD map has i-quant slots"; exit 1; }
DPL=$("$PY" -c "import json,statistics as s;d=json.load(open('$MAP'));v=[len(x) for x in d.values() if isinstance(x,list)];print(int(s.mean(v)) if v else -1)")
L "fk16 drop map dropped/layer=$DPL (expect 30 -> 98e)"; [ "$DPL" = "30" ] || { L "FATAL not 98e"; exit 1; }
FREEG=$(df -PB1G "$WORK" | awk 'NR==2{print $4+0}')
L "disk free=${FREEG}G (need ~120G)"; [ "${FREEG:-0}" -lt 90 ] && { L "FATAL low disk"; exit 1; }

# [1] expert_drop (QAT base + fk16 map)
if [ -f "$DIR/model.safetensors.index.json" ]; then L "[1] expert_drop dir exists, skip"; else
  L "[1] expert_drop QAT_base -> $DIR"
  "$PY" "$SCR/expert_drop.py" --source-dir "$QAT_BASE" --drop-map "$MAP" --output-dir "$DIR" 2>&1 | tail -6
  [ -f "$DIR/model.safetensors.index.json" ] || { L "FATAL expert_drop failed"; exit 1; }
fi
for aux in preprocessor_config.json processor_config.json generation_config.json tokenizer.json tokenizer_config.json config.json; do
  if [ ! -f "$DIR/$aux" ] && [ -f "$AUX/$aux" ]; then cp -n "$AUX/$aux" "$DIR/$aux"; L "  aux $aux"; fi
done

# [2] shared alpha=1.2 (published recipe — matches cohort bf16)
if [ ! -f "$DIR/.shared_applied" ]; then
  L "[2] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
  "$PY" "$SCR/router_shared_upweight.py" --model-dir "$DIR" --alpha 1.2 --target mlp.down_proj.weight 2>&1 | tail -4
  rm -f "$DIR"/*.pre_shared_upweight; touch "$DIR/.shared_applied"
else L "[2] .shared_applied present, skip"; fi

# [3] convert -> F16 (KEPT); drop the bf16 dir afterwards (only F16 needed downstream)
if gguf_ok "$F16"; then L "[3] F16 exists, skip"; else
  L "[3] convert -> F16"
  "$PY" "$CONVERT" "$DIR" --outfile "$F16" --outtype f16 2>&1 | tail -3
  n=$("$PY" -c "from gguf.gguf_reader import GGUFReader;print(len(GGUFReader('$F16').tensors))" 2>/dev/null)
  L "  F16 tensors=$n"; [ "${n:-0}" -lt 600 ] && { L "FATAL too few tensors ($n)"; exit 1; }
fi
[ -d "$DIR" ] && { rm -rf "$DIR"; L "  removed bf16 work dir (F16 retained)"; }

# [4] qat-Q4_0 (NO imatrix — CPU)
if gguf_ok "$Q40"; then L "[4] qat-Q4_0 exists, skip"; else
  L "[4] llama-quantize Q4_0 (no imatrix) -> qat-Q4_0"
  "$BIN/llama-quantize" "$F16" "$Q40" Q4_0 2>&1 | tail -3
  gguf_ok "$Q40" || { L "FATAL qat-Q4_0 bad header"; exit 1; }
fi
L "  qat-Q4_0 = $(du -h "$Q40"|cut -f1)"

# [5] qat imatrix (GPU) — wait until the plain loop gate is done (no GPU contention)
if [ -s "$IMAT" ]; then L "[5] qat imatrix exists, skip"; else
  L "[5] waiting for plain loop gate to finish before grabbing a GPU ..."
  for i in $(seq 1 720); do
    miss=""; for T in $PLAIN_NEED; do [ -f "$DONE/$T.done" ] || miss="$miss$T "; done
    [ -z "$miss" ] && { L "  plain gate done — proceeding"; break; }
    sleep 30
  done
  GPU=-1
  for i in $(seq 1 120); do
    for g in 0 1; do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
      [ -n "$m" ] && [ "$m" -lt 3000 ] && { GPU=$g; break; }
    done
    [ "$GPU" -ge 0 ] && break; sleep 20
  done
  [ "$GPU" -ge 0 ] || { L "FATAL no GPU freed for imatrix"; exit 1; }
  L "[5] llama-imatrix on GPU$GPU (calib_both) ..."
  CUDA_VISIBLE_DEVICES=$GPU "$BIN/llama-imatrix" -m "$F16" -f "$CALIB" -o "$IMAT" -ngl 99 2>&1 | tail -8
  [ -s "$IMAT" ] || { L "FATAL qat imatrix failed"; exit 1; }
fi
L "  qat imatrix = $(du -h "$IMAT"|cut -f1)"
cp -n "$IMAT" "$GG/imatrix-qat.dat" 2>/dev/null && L "  archived imatrix-qat.dat next to quants" || true

# [6] CD-qat-Q4_K_M (CPU)
if gguf_ok "$CDQAT"; then L "[6] CD-qat-Q4_K_M exists, skip"; else
  L "[6] llama-quantize Q4_K_M + CD map -> CD-qat-Q4_K_M"
  "$BIN/llama-quantize" --imatrix "$IMAT" --tensor-type-file "$CD" "$F16" "$CDQAT" Q4_K_M 2>&1 | tail -3
  gguf_ok "$CDQAT" || { L "FATAL CD-qat bad header"; exit 1; }
fi
L "  CD-qat-Q4_K_M = $(du -h "$CDQAT"|cut -f1)"

L "=== DONE: qat-Q4_0=$(du -h "$Q40"|cut -f1)  CD-qat-Q4_K_M=$(du -h "$CDQAT"|cut -f1) ==="
L "###### STD16_QAT_TIERS_DONE ######"
