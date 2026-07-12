#!/usr/bin/env bash
# build_nofold_gate.sh — T222 DECISIVE: does the SELECTION-ONLY (no-fold) student loop?
#
# The fold A/B proved: (1) no fold recovers GPQA capability (avg 99 = noDERN 99 = LS 104, all
# vs 146 ceiling — fold only reshuffles), (2) the average fold's survivor-BLUR is what buys the
# 0/48 anti-loop, inseparable from survivor corruption, (3) LS-fold (pure gate_up, blended down
# = INCOHERENT) loops 13/9. The one cell never measured: the fkbroad student with NO fold at all
# (pure-survivor experts, gate_up AND down both pristine = COHERENT). This decides the recipe:
#   no-fold ~0/48  -> COHERENCE matters, not blur; DERN is baggage; drop it, capability via
#                     selection (extcap) + decode/norm loop guard.
#   no-fold ~13/9  -> pure-survivor sharpness loops; the average blur is load-bearing medicine;
#                     keep average fold + tune selection (extcap2 = avg-fold + science = 67.7/2-loops).
#
# Build = build_v8_lsfold.sh stages 1-3 MINUS the DERN fold (stage 4). Same selection (fkbroad
# drop map), same shared a=1.2, same calib_both imatrix, same imat-Q6 tier, same 48-seed gate.
# GPU0, llama.cpp-latest. PID-kill only.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
LCPP=/mnt/sdc/ml/llama.cpp-latest
SFT=/mnt/sdc/ml/sft_heal
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
DROP=/srv/ml/scripts/v8coder_fkbroad_drop_map.json
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
AL=/srv/ml/agentic_loop
GPU=0

STU=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-nofold-it   # selection-only student (NO fold)
F16=$SFT/fkbroad-nofold-F16.gguf
IMAT=$SFT/fkbroad-nofold-imatrix.dat                  # PRESERVED
Q6=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-nofold-imat-Q6_K.gguf
GRES=$AL/results/fkbroad-nofold-imatQ6_minp48.json
ts(){ date '+%T %Z'; }
echo "==================== build NO-FOLD student + loop gate $(ts) ===================="

# ── preflight ─────────────────────────────────────────────
for f in "$SRC/config.json" "$DROP" "$CALIB" "$SCR/expert_drop.py" \
         "$RECIPE/router_shared_upweight.py" "$LCPP/convert_hf_to_gguf.py" \
         "$LCPP/build/bin/llama-imatrix" "$LCPP/build/bin/llama-quantize" "$GATE"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
free=$(df --output=avail -BG "$SFT" | tail -1 | tr -dc '0-9')
echo "[preflight $(ts)] ${free}G free on $SFT"
[ "${free:-0}" -lt 90 ] && { echo "FATAL <90G free — refusing (peak ~115G); reclaim first"; exit 9; }

# ── 1. expert_drop (fkbroad selection) ───────────────────
if [ ! -f "$Q6" ] && [ ! -f "$F16" ]; then
  if [ ! -f "$STU/model.safetensors.index.json" ] && [ ! -f "$STU/model.safetensors" ]; then
    echo "[1 $(ts)] expert_drop(fkbroad) -> $STU"
    "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$STU" \
      || { echo "FATAL expert_drop"; exit 3; }
    [ -f "$STU/tokenizer.json" ] || { echo "FATAL tokenizer.json not copied"; exit 3; }
  else echo "[1] $STU exists, skip"; fi

  # ── 2. router_shared_upweight (a=1.2) — IDENTICAL to v8/lsfold ──
  if [ ! -f "$STU/.shared_applied" ]; then
    echo "[2 $(ts)] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
    "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$STU" \
      --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared_upweight"; exit 4; }
    touch "$STU/.shared_applied"
  else echo "[2] .shared_applied exists, skip"; fi
  # NO stage 3/4: no keepmeta, NO DERN fold — this IS the experiment.
fi

# ── 5. convert F16, then drop the student bf16 (reproducible; tight disk) ──
if [ ! -f "$Q6" ]; then
  [ -f "$F16" ] || { echo "[5 $(ts)] convert -> $F16";
    "$PY" "$LCPP/convert_hf_to_gguf.py" "$STU" --outfile "$F16" --outtype f16 \
      || { echo "FATAL convert"; exit 8; }; }
  if [ -d "$STU" ]; then echo "[5 $(ts)] convert done; removing student bf16 $STU"; rm -rf "$STU"; fi
fi

# ── 6. model-specific imatrix (calib_both, 128, ngl99) — PRESERVED ──
if [ ! -f "$IMAT" ] && [ ! -f "$Q6" ]; then
  echo "[6 $(ts)] llama-imatrix -ngl 99 --chunks 128 (GPU$GPU) -> $IMAT"
  CUDA_VISIBLE_DEVICES=$GPU "$LCPP/build/bin/llama-imatrix" \
    -m "$F16" -f "$CALIB" -o "$IMAT" --chunks 128 -ngl 99 \
    > "$SFT/fkbroad_nofold_imatrix_build.log" 2>&1 \
    || { echo "FATAL imatrix"; tail -25 "$SFT/fkbroad_nofold_imatrix_build.log"; exit 9; }
fi
[ -f "$IMAT" ] && echo "[6 $(ts)] imatrix.dat $(stat -c%s "$IMAT" 2>/dev/null) bytes (PRESERVED)"

# ── 7. imat-Q6_K, then drop F16 ──────────────────────────
[ -f "$Q6" ] || { echo "[7 $(ts)] quantize imat-Q6_K -> $Q6";
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 \
    || { echo "FATAL quant Q6"; exit 10; }; }
magic=$("$PY" -c "import sys;print(open('$Q6','rb').read(4).decode('latin1'))" 2>/dev/null)
[ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF header"; exit 10; }
echo "[7 $(ts)] Q6 $(du -h "$Q6"|cut -f1); dropping F16"; rm -f "$F16"

# ── 8. DECISIVE loop gate imat-Q6 {0.9,0.8}x48 ───────────
echo "[8 $(ts)] loop gate NO-FOLD imat-Q6 GPU$GPU:8262"
echo "[8 $(ts)] compare: avg-fold 0/48,0/48 | LS-fold 13/48,9/48 | no-fold = ?"
bash "$GATE" "$Q6" "$GPU" 8262 "$GRES" nofold-imatQ6 || echo "[8] WARN gate rc=$?"

echo "[9 $(ts)] === NO-FOLD GATE DONE ==="
[ -f "$GRES" ] && "$PY" -c "import json;d=json.load(open('$GRES'));[print(' ',r['config'],'loops',r['loops'],'/',r['seeds']) for r in d['results']]" 2>/dev/null
echo "imatrix preserved: $IMAT ; Q6: $Q6"
echo "###### NOFOLD_GATE_DONE $(ts) ######"
