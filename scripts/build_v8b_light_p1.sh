#!/usr/bin/env bash
# build_v8b_light_p1.sh — v8b-light = fkbroad recipe with generic_science 1->1.5 +
# generic_multilingual 0->1 (class-weights 1 1 3 1.5 1 1 0 0 2), SAME 30 code/LCB
# force-keep pins. Restores 91 science/ml specialists at cost of 14 code specialists
# (v8b_pick). Everything else IDENTICAL to v8 soft2: shared a=1.2, DERN Eq.11
# --assign-topk 2, eog_corpus_solar, calib_both imatrix, imat-Q6 (llama.cpp-latest,
# matching how v8 was built). Phase 1 = build + 48-seed agentic gate on b9700.
# Phase 2 (separate) = full GPQA/ARC/IFEval + HE+/MPE/LCB eval on b9700.
# Self-scheduling, resumable, PID-kill only.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
DERN=$SCR/redist_dern_eq11.py
LCPP=/mnt/sdc/ml/llama.cpp-latest          # build binary (matches v8)
SFT=/mnt/sdc/ml/sft_heal
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it # 128e teacher + tokenizer
DROP=/srv/ml/scripts/v8b_light_drop_map.json
CORPUS=$SFT/eog_corpus_solar.jsonl
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt
AL=/srv/ml/agentic_loop
GATE_SRC=/srv/ml/scripts/gate_sweep48_minp_p.sh
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh   # b9700 variant (created below)

V8B=$SFT/gemma-4-A4B-98e-v7-coder-v8b-light-it
KEEPMETA=$SFT/v8b_light_keepmeta.json
COMBO=$SFT/gemma-4-A4B-98e-v7-coder-v8b-light-soft2-it
F16=$SFT/v8b-light-soft2-F16.gguf
IMAT=$SFT/v8b-light-soft2-imatrix.dat       # PRESERVED (mandatory)
Q6=$SFT/gemma-4-A4B-98e-v7-coder-v8b-light-soft2-imat-Q6_K.gguf
GRES=$AL/results/v8b-light-soft2-imatQ6_minp48.json
NAME=v8b-light-soft2-imatq6
PORT=8192
ts(){ date '+%T %Z'; }
echo "==================== build v8b-light soft2 (Phase 1: build+gate) $(ts) ===================="

# preflight
for f in "$SRC/config.json" "$DROP" "$CORPUS" "$CALIB" "$SCR/expert_drop.py" \
         "$RECIPE/router_shared_upweight.py" "$SCR/redist_prep_v7coder.py" "$DERN" \
         "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-imatrix" \
         "$LCPP/build/bin/llama-quantize" "$GATE_SRC" \
         /mnt/sdc/ml/llama.cpp-b9700/build/bin/llama-server; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
grep -q -- "--assign-topk" "$DERN" || { echo "FATAL redist not patched"; exit 2; }
# b9700 gate variant (only the LS= line differs)
[ -f "$GATE" ] || sed 's#/mnt/sdc/ml/llama.cpp-latest/build/bin/llama-server#/mnt/sdc/ml/llama.cpp-b9700/build/bin/llama-server#' "$GATE_SRC" > "$GATE"
echo "[preflight $(ts)] disk:"; df -h "$SFT" | tail -1

# 1. expert_drop (v8b-light keep set)
if [ ! -f "$COMBO/model.safetensors.index.json" ] && [ ! -f "$COMBO/model.safetensors" ]; then
  if [ ! -f "$V8B/model.safetensors.index.json" ] && [ ! -f "$V8B/model.safetensors" ]; then
    echo "[1 $(ts)] expert_drop(v8b-light) -> $V8B"
    "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$V8B" \
      || { echo "FATAL expert_drop"; exit 3; }
    [ -f "$V8B/tokenizer.json" ] || { echo "FATAL tokenizer.json not copied"; exit 3; }
  else echo "[1] $V8B exists, skip"; fi
  # 2. router_shared_upweight a=1.2
  if [ ! -f "$V8B/.shared_applied" ]; then
    echo "[2 $(ts)] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
    "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$V8B" \
      --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared_upweight"; exit 4; }
    touch "$V8B/.shared_applied"
  else echo "[2] .shared_applied exists, skip"; fi
  # 3. keep-meta
  if [ ! -f "$KEEPMETA" ]; then
    echo "[3 $(ts)] redist_prep_v7coder -> $KEEPMETA"
    "$PY" "$SCR/redist_prep_v7coder.py" "$V8B" "$DROP" "$KEEPMETA" \
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

# 4. DERN Eq.11 soft-top-2 fold
if [ ! -f "$COMBO/model.safetensors.index.json" ] && [ ! -f "$COMBO/model.safetensors" ]; then
  echo "[4 $(ts)] redist_dern_eq11 --assign-topk 2 -> $COMBO (GPU$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$DERN" --teacher "$SRC" --student "$V8B" --keep-meta "$KEEPMETA" \
      --freq-corpus "$CORPUS" --seq-corpus "$CORPUS" --out "$COMBO" --device cuda:0 \
      --assign-topk 2 || { echo "FATAL redist_dern_eq11"; exit 7; }
  echo "[4 $(ts)] DERN done; removing intermediate student $V8B"
  rm -rf "$V8B"
else echo "[4] $COMBO exists, skip redist"; fi

# 5. convert F16
[ -f "$F16" ] || { echo "[5 $(ts)] convert -> $F16";
  "$PY" "$LCPP/convert_hf_to_gguf.py" "$COMBO" --outfile "$F16" --outtype f16 \
    || { echo "FATAL convert"; exit 8; }; }

# 6. model-specific imatrix (calib_both, 128 chunks, ngl 99) — PRESERVED
if [ ! -f "$IMAT" ]; then
  echo "[6 $(ts)] llama-imatrix -ngl 99 --chunks 128 (GPU$GPU) -> $IMAT"
  CUDA_VISIBLE_DEVICES=$GPU "$LCPP/build/bin/llama-imatrix" \
    -m "$F16" -f "$CALIB" -o "$IMAT" --chunks 128 -ngl 99 \
    > "$SFT/v8b_light_imatrix_build.log" 2>&1 \
    || { echo "FATAL imatrix"; tail -25 "$SFT/v8b_light_imatrix_build.log"; exit 9; }
fi
echo "[6 $(ts)] imatrix.dat $(stat -c%s "$IMAT" 2>/dev/null) bytes"

# 7. imat-Q6_K, then drop F16
[ -f "$Q6" ] || { echo "[7 $(ts)] quantize imat-Q6_K -> $Q6";
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 \
    || { echo "FATAL quant"; exit 10; }; }
magic=$("$PY" -c "import sys;print(open('$Q6','rb').read(4).decode('latin1'))" 2>/dev/null)
[ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF header"; exit 10; }
rm -f "$F16"
echo "[7 $(ts)] Q6 ready: $(ls -la "$Q6" | sed 's/  */ /g')"

# 8. 48-seed agentic loop gate on b9700 vs v8 (0/48,0/48)
echo "[8 $(ts)] loop gate $NAME {t0.9,t0.8} GPU$GPU:$PORT on b9700 (v8 ref: 0/48,0/48)"
bash "$GATE" "$Q6" "$GPU" "$PORT" "$GRES" "$NAME" || echo "[8] WARN gate rc=$?"

echo "==================== PHASE 1 DONE $(ts) ===================="
echo "imatrix preserved: $IMAT"; echo "combo bf16: $COMBO"; echo "imat-Q6: $Q6"
echo "gate result: $GRES"
[ -f "$GRES" ] && "$PY" -c "import json;d=json.load(open('$GRES'));[print('  ',r['config'],'loops',r.get('loops'),'/',r['seeds']) for r in d['results']]" 2>/dev/null
