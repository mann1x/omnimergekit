#!/usr/bin/env bash
# build_v8b_safe_pes_p1.sh — targeted per-pin PES on the v8b-safe soft2 combo.
# GGUF-safe per-expert mixing boost: bake alpha into ONLY the 30 agentic-EOG
# terminator pins' routed down_proj (experts.down_proj[idx] -> ffn_down_exps),
# the math-identical-to-per_expert_scale lever that actually survives GGUF
# conversion (router.per_expert_scale is dropped by convert_hf_to_gguf).
# Purpose: reclaim the router mixing mass the restored science experts steal
# from the pins, closing v8b-safe's residual agentic-loop gap (3/48,4/48) toward
# 0/48 WITHOUT dropping any of the 85 restored science/multilingual experts.
# PURE post-hoc weight edit on the EXISTING combo — no re-prune, no re-DERN,
# no re-shared-upweight. Everything downstream IDENTICAL to v8b-safe: calib_both
# imatrix (128 chunks, ngl 99) -> imat-Q6 (llama.cpp-latest) -> 48-seed b9700
# agentic gate at vendor_minp_rep {t0.9,t0.8}. Self-scheduling, resumable,
# PID-kill only. Hard bar 0/48 (v8 ref 0/48,0/48; v8b-safe ref 3/48,4/48).
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
LCPP=/mnt/sdc/ml/llama.cpp-latest          # build binary (matches v8 / v8b-safe)
SFT=/mnt/sdc/ml/sft_heal
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt
AL=/srv/ml/agentic_loop
GATE_SRC=/srv/ml/scripts/gate_sweep48_minp_p.sh
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh
BAKE=/mnt/sdc/ml/v8b_pes/router_pin_downproj_bake.py
KEEPMETA=$SFT/v8b_safe_keepmeta.json
PINS="0:2,0:9,0:10,0:86,0:93,1:44,1:92,1:95,2:4,2:113,7:66,8:80,8:114,9:110,10:12,11:8,11:82,11:118,12:6,12:38,13:82,14:45,14:102,15:39,15:47,15:99,17:57,17:71,21:10,23:96"

ALPHA=1.3
PESTAG=pes13                                # alpha 1.3 -> pes13 (bump for new rungs)
SRC_COMBO=$SFT/gemma-4-A4B-98e-v7-coder-v8b-safe-soft2-it
PES_COMBO=$SFT/gemma-4-A4B-98e-v7-coder-v8b-safe-${PESTAG}-soft2-it
F16=$SFT/v8b-safe-${PESTAG}-soft2-F16.gguf
IMAT=$SFT/v8b-safe-${PESTAG}-soft2-imatrix.dat     # PRESERVED (mandatory)
Q6=$SFT/gemma-4-A4B-98e-v7-coder-v8b-safe-${PESTAG}-soft2-imat-Q6_K.gguf
GRES=$AL/results/v8b-safe-${PESTAG}-soft2-imatQ6_minp48.json
NAME=v8b-safe-${PESTAG}-soft2-imatq6
PORT=8196
ts(){ date '+%T %Z'; }
echo "==================== build v8b-safe ${PESTAG} (alpha=${ALPHA}) $(ts) ===================="

# preflight
for f in "$SRC_COMBO/model.safetensors" "$SRC_COMBO/config.json" "$KEEPMETA" "$BAKE" \
         "$CALIB" "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-imatrix" \
         "$LCPP/build/bin/llama-quantize" "$GATE_SRC" \
         /mnt/sdc/ml/llama.cpp-b9700/build/bin/llama-server; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
# b9700 gate variant (only the llama-server path differs); reuse if already made
[ -f "$GATE" ] || sed 's#/mnt/sdc/ml/llama.cpp-latest/build/bin/llama-server#/mnt/sdc/ml/llama.cpp-b9700/build/bin/llama-server#' "$GATE_SRC" > "$GATE"
echo "[preflight $(ts)] disk:"; df -h "$SFT" | tail -1

# 1. build the PES combo dir (sidecars + baked single-file model.safetensors)
if [ ! -f "$PES_COMBO/model.safetensors" ]; then
  echo "[1 $(ts)] stage sidecars -> $PES_COMBO (exclude model.safetensors + dead backups)"
  mkdir -p "$PES_COMBO"
  rsync -a --exclude='model.safetensors' --exclude='*.pre_shared_upweight' \
        --exclude='*.pre_per_expert_rescale' "$SRC_COMBO"/ "$PES_COMBO"/ \
    || { echo "FATAL sidecar rsync"; exit 3; }
  echo "[1 $(ts)] bake alpha=$ALPHA into 30 pin slabs of experts.down_proj"
  "$PY" "$BAKE" --in-model "$SRC_COMBO/model.safetensors" \
        --out-model "$PES_COMBO/model.safetensors" \
        --keep-meta "$KEEPMETA" --pins "$PINS" --alpha "$ALPHA" \
    || { echo "FATAL bake"; exit 3; }
  sz=$(stat -c%s "$PES_COMBO/model.safetensors" 2>/dev/null)
  src_sz=$(stat -c%s "$SRC_COMBO/model.safetensors")
  [ "$sz" = "$src_sz" ] || { echo "FATAL size mismatch baked=$sz src=$src_sz"; exit 3; }
  echo "[1 $(ts)] PES combo ready: model.safetensors $sz bytes (== src)"
else echo "[1] $PES_COMBO/model.safetensors exists, skip bake"; fi

# acquire first free GPU (<2000 MiB), up to 4h
GPU=""
for i in $(seq 1 240); do
  for g in 0 1; do
    U=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
    [ "${U:-99999}" -lt 2000 ] && { GPU=$g; break; }
  done
  [ -n "$GPU" ] && break
  echo "[acquire $(ts)] both GPUs busy, wait 60s ($i/240)"; sleep 60
done
[ -n "$GPU" ] || { echo "FATAL no free GPU after 4h"; exit 6; }
echo "[acquire $(ts)] using GPU$GPU"

# 2. convert F16
[ -f "$F16" ] || { echo "[2 $(ts)] convert -> $F16";
  "$PY" "$LCPP/convert_hf_to_gguf.py" "$PES_COMBO" --outfile "$F16" --outtype f16 \
    || { echo "FATAL convert"; exit 8; }; }

# 3. model-specific imatrix (calib_both, 128 chunks, ngl 99) — PRESERVED
if [ ! -f "$IMAT" ]; then
  echo "[3 $(ts)] llama-imatrix -ngl 99 --chunks 128 (GPU$GPU) -> $IMAT"
  CUDA_VISIBLE_DEVICES=$GPU "$LCPP/build/bin/llama-imatrix" \
    -m "$F16" -f "$CALIB" -o "$IMAT" --chunks 128 -ngl 99 \
    > "$SFT/v8b_safe_${PESTAG}_imatrix_build.log" 2>&1 \
    || { echo "FATAL imatrix"; tail -25 "$SFT/v8b_safe_${PESTAG}_imatrix_build.log"; exit 9; }
fi
echo "[3 $(ts)] imatrix.dat $(stat -c%s "$IMAT" 2>/dev/null) bytes"

# 4. imat-Q6_K, then drop F16
[ -f "$Q6" ] || { echo "[4 $(ts)] quantize imat-Q6_K -> $Q6";
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 \
    || { echo "FATAL quant"; exit 10; }; }
magic=$("$PY" -c "import sys;print(open('$Q6','rb').read(4).decode('latin1'))" 2>/dev/null)
[ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF header"; exit 10; }
rm -f "$F16"
echo "[4 $(ts)] Q6 ready: $(ls -la "$Q6" | sed 's/  */ /g')"

# 5. 48-seed agentic loop gate on b9700 (v8 ref 0/48,0/48 | v8b-safe ref 3/48,4/48)
echo "[5 $(ts)] loop gate $NAME {t0.9,t0.8} GPU$GPU:$PORT on b9700"
bash "$GATE" "$Q6" "$GPU" "$PORT" "$GRES" "$NAME" || echo "[5] WARN gate rc=$?"

echo "==================== ${PESTAG} DONE $(ts) ===================="
echo "imatrix preserved: $IMAT"; echo "PES combo bf16: $PES_COMBO"; echo "imat-Q6: $Q6"
echo "gate result: $GRES"
[ -f "$GRES" ] && "$PY" -c "import json;d=json.load(open('$GRES'));[print('  ',r['config'],'loops',r.get('loops'),'/',r['seeds']) for r in d['results']]" 2>/dev/null
