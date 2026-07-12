#!/usr/bin/env bash
# build_creative_sacrifice.sh — creative-sacrifice = cap2 science set, but every
# eviction routed to generic_creative (137) + 7 weak logic/math, 0 multilingual/
# code/HE+ touched. Hypothesis: cap2 loops (~1) + cap2 GPQA (~67.7) + MPE holds
# (~89) because creative is not a hard-hold bench. Single-variable vs cap2: only
# the victim class differs. Recipe IDENTICAL to v8/cap2 soft2: shared a=1.2,
# DERN Eq.11 --assign-topk 2, eog_corpus_solar, calib_both imatrix, imat-Q6
# (llama.cpp-latest). Phase 1 = build + 48-seed agentic loop gate on b9700.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
DERN=$SCR/redist_dern_eq11.py
LCPP=/mnt/sdc/ml/llama.cpp-latest
SFT=/mnt/sdc/ml/sft_heal
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
DROP=/mnt/sdc/ml/gpqa_dissect/v8b_sweep/creative_sacrifice_drop_map.json
CORPUS=$SFT/eog_corpus_solar.jsonl
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt
AL=/srv/ml/agentic_loop
GATE_SRC=/srv/ml/scripts/gate_sweep48_minp_p.sh
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh

STU=$SFT/gemma-4-A4B-98e-creative-sacrifice-it
KEEPMETA=$SFT/creative_sacrifice_keepmeta.json
COMBO=$SFT/gemma-4-A4B-98e-creative-sacrifice-soft2-it
F16=$SFT/creative-sacrifice-soft2-F16.gguf
IMAT=$SFT/creative-sacrifice-soft2-imatrix.dat
Q6=$SFT/gemma-4-A4B-98e-creative-sacrifice-soft2-imat-Q6_K.gguf
GRES=$AL/results/creative-sacrifice-soft2-imatQ6_minp48.json
NAME=creative-sacrifice-soft2-imatq6
PORT=8196
ts(){ date '+%T %Z'; }
echo "==================== build creative-sacrifice soft2 (build+gate) $(ts) ===================="

for f in "$SRC/config.json" "$DROP" "$CORPUS" "$CALIB" "$SCR/expert_drop.py" \
         "$RECIPE/router_shared_upweight.py" "$SCR/redist_prep_v7coder.py" "$DERN" \
         "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-imatrix" \
         "$LCPP/build/bin/llama-quantize" "$GATE_SRC" \
         /mnt/sdc/ml/llama.cpp-b9700/build/bin/llama-server; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
grep -q -- "--assign-topk" "$DERN" || { echo "FATAL redist not patched"; exit 2; }
[ -f "$GATE" ] || sed 's#/mnt/sdc/ml/llama.cpp-latest/build/bin/llama-server#/mnt/sdc/ml/llama.cpp-b9700/build/bin/llama-server#' "$GATE_SRC" > "$GATE"
echo "[preflight $(ts)] disk:"; df -h "$SFT" | tail -1

# 1. expert_drop
if [ ! -f "$COMBO/model.safetensors.index.json" ] && [ ! -f "$COMBO/model.safetensors" ]; then
  if [ ! -f "$STU/model.safetensors.index.json" ] && [ ! -f "$STU/model.safetensors" ]; then
    echo "[1 $(ts)] expert_drop(creative-sacrifice) -> $STU"
    "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$STU" \
      || { echo "FATAL expert_drop"; exit 3; }
    [ -f "$STU/tokenizer.json" ] || { echo "FATAL tokenizer.json not copied"; exit 3; }
  else echo "[1] $STU exists, skip"; fi
  # 2. shared a=1.2
  if [ ! -f "$STU/.shared_applied" ]; then
    echo "[2 $(ts)] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
    "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$STU" \
      --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared_upweight"; exit 4; }
    touch "$STU/.shared_applied"
  else echo "[2] .shared_applied exists, skip"; fi
  # 3. keep-meta
  if [ ! -f "$KEEPMETA" ]; then
    echo "[3 $(ts)] redist_prep_v7coder -> $KEEPMETA"
    "$PY" "$SCR/redist_prep_v7coder.py" "$STU" "$DROP" "$KEEPMETA" \
      || { echo "FATAL keepmeta prep"; exit 5; }
  fi
else echo "[1-3] COMBO exists, skip student build"; fi

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

# 4. DERN Eq.11 soft-top-2
if [ ! -f "$COMBO/model.safetensors.index.json" ] && [ ! -f "$COMBO/model.safetensors" ]; then
  echo "[4 $(ts)] redist_dern_eq11 --assign-topk 2 -> $COMBO (GPU$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$DERN" --teacher "$SRC" --student "$STU" --keep-meta "$KEEPMETA" \
      --freq-corpus "$CORPUS" --seq-corpus "$CORPUS" --out "$COMBO" --device cuda:0 \
      --assign-topk 2 || { echo "FATAL redist_dern_eq11"; exit 7; }
  echo "[4 $(ts)] DERN done; removing intermediate student $STU"
  rm -rf "$STU"
else echo "[4] $COMBO exists, skip redist"; fi

# 5. convert F16
[ -f "$F16" ] || { echo "[5 $(ts)] convert -> $F16";
  "$PY" "$LCPP/convert_hf_to_gguf.py" "$COMBO" --outfile "$F16" --outtype f16 \
    || { echo "FATAL convert"; exit 8; }; }

# 6. imatrix (calib_both, 128 chunks, ngl 99) — PRESERVED
if [ ! -f "$IMAT" ]; then
  echo "[6 $(ts)] llama-imatrix -ngl 99 --chunks 128 (GPU$GPU) -> $IMAT"
  CUDA_VISIBLE_DEVICES=$GPU "$LCPP/build/bin/llama-imatrix" \
    -m "$F16" -f "$CALIB" -o "$IMAT" --chunks 128 -ngl 99 \
    > "$SFT/creative_sacrifice_imatrix_build.log" 2>&1 \
    || { echo "FATAL imatrix"; tail -25 "$SFT/creative_sacrifice_imatrix_build.log"; exit 9; }
fi
echo "[6 $(ts)] imatrix.dat $(stat -c%s "$IMAT" 2>/dev/null) bytes"

# 7. imat-Q6_K, drop F16
[ -f "$Q6" ] || { echo "[7 $(ts)] quantize imat-Q6_K -> $Q6";
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 \
    || { echo "FATAL quant"; exit 10; }; }
magic=$("$PY" -c "import sys;print(open('$Q6','rb').read(4).decode('latin1'))" 2>/dev/null)
[ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF header"; exit 10; }
rm -f "$F16"
echo "[7 $(ts)] Q6 ready: $(ls -la "$Q6" | sed 's/  */ /g')"

# 8. 48-seed agentic loop gate on b9700 (cap2 ref: 1/48, 0/48)
echo "[8 $(ts)] loop gate $NAME {t0.9,t0.8} GPU$GPU:$PORT on b9700 (cap2 ref 1/48,0/48)"
bash "$GATE" "$Q6" "$GPU" "$PORT" "$GRES" "$NAME" || echo "[8] WARN gate rc=$?"

echo "==================== BUILD+GATE DONE $(ts) ===================="
echo "imatrix preserved: $IMAT"; echo "combo bf16: $COMBO"; echo "imat-Q6: $Q6"
[ -f "$GRES" ] && "$PY" -c "import json;d=json.load(open('$GRES'));[print('  ',r['config'],'loops',r.get('loops'),'/',r['seeds']) for r in d['results']]" 2>/dev/null
