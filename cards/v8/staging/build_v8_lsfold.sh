#!/usr/bin/env bash
# build_v8_lsfold.sh — T222 LS-fold A/B: single-variable swap of the DERN fold math.
#
# IDENTICAL to build_v8_fkbroad_soft2_combo.sh (the shipping v8) EXCEPT step 4 points DERN
# at redist_dern_eq11_fold.py with --fold-mode mergemoe. Selection (fkbroad force-keep),
# shared a=1.2, soft top-2 assignment, freq weights, eog_corpus_solar, calib_both imatrix
# are byte-for-byte the same — so the only thing that differs vs the average-fold v8 is the
# per-survivor fold:
#   average  (shipping v8): survivor gate_up <- freq-weighted AVERAGE (blends feature dir),
#                           Eq.11 rescales output norm.
#   mergemoe (this build):  survivor gate_up KEPT EXACTLY; down LS-solved to reproduce the
#                           SAME freq-weighted blend target in the survivor's feature subspace.
# Numeric guard (smoke_fold_equiv.py) already PROVED --fold-mode average == dern11 bitwise,
# and --fold-mode mergemoe keeps gate_up bitwise-identical + output-norm ratio ~0.98.
#
# Builds BOTH the decisive imat-Q6 (must stay 0/48 like v8) AND the corruption-exposing
# no-imat Q4_0 (v8-average loops 9/48,12/48 — does survivor preservation reduce it?), runs
# the 48-seed loop gate on each. GPU0, llama.cpp-latest. GPQA/gap + HE+/MPE run separately
# on GPU1 (eval_v8_lsfold.sh, b9700). PID-kill only; gates own their servers.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
DERN=/srv/ml/repos/omnimergekit/scripts/redist_dern_eq11_fold.py   # the swap
LCPP=/mnt/sdc/ml/llama.cpp-latest
SFT=/mnt/sdc/ml/sft_heal
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
DROP=/srv/ml/scripts/v8coder_fkbroad_drop_map.json
CORPUS=$SFT/eog_corpus_solar.jsonl
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
AL=/srv/ml/agentic_loop
GPU=0

FKB=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-lsfold-stage-it   # student (regen; deleted after DERN)
KEEPMETA=$SFT/v7coder_fkbroad_keepmeta.json                 # SHARED with soft2 (exists)
COMBO=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-lsfold-it       # lsfold bf16 (KEPT)
F16=$SFT/fkbroad-lsfold-F16.gguf
IMAT=$SFT/fkbroad-lsfold-imatrix.dat                        # PRESERVED (mandatory)
Q6=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-lsfold-imat-Q6_K.gguf
Q40=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-lsfold-noimat-Q4_0.gguf
GRES_Q6=$AL/results/fkbroad-lsfold-imatQ6_minp48.json
GRES_Q40=$AL/results/fkbroad-lsfold-noimatQ40_minp48.json
ts(){ date '+%T %Z'; }
echo "==================== build v8 LS-FOLD (mergemoe) $(ts) ===================="

# ── preflight ─────────────────────────────────────────────
for f in "$SRC/config.json" "$DROP" "$CORPUS" "$CALIB" "$SCR/expert_drop.py" \
         "$RECIPE/router_shared_upweight.py" "$SCR/redist_prep_v7coder.py" "$DERN" \
         "$KEEPMETA" "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-imatrix" \
         "$LCPP/build/bin/llama-quantize" "$GATE"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
grep -q -- "--fold-mode" "$DERN" || { echo "FATAL fold script not patched (no --fold-mode)"; exit 2; }
free=$(df --output=avail -BG "$SFT" | tail -1 | tr -dc '0-9')
echo "[preflight $(ts)] ${free}G free on $SFT"
[ "${free:-0}" -lt 90 ] && { echo "FATAL <90G free — refusing (peak ~80G); reclaim first"; exit 9; }

# ── stages 1-3: regen fkbroad student bf16 (skip if COMBO already folded) ──
if [ ! -f "$COMBO/model.safetensors.index.json" ] && [ ! -f "$COMBO/model.safetensors" ]; then
  if [ ! -f "$FKB/model.safetensors.index.json" ] && [ ! -f "$FKB/model.safetensors" ]; then
    echo "[1 $(ts)] expert_drop(fkbroad) -> $FKB"
    "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$FKB" \
      || { echo "FATAL expert_drop"; exit 3; }
    [ -f "$FKB/tokenizer.json" ] || { echo "FATAL tokenizer.json not copied"; exit 3; }
  else echo "[1] $FKB exists, skip"; fi
  if [ ! -f "$FKB/.shared_applied" ]; then
    echo "[2 $(ts)] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
    "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$FKB" \
      --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared_upweight"; exit 4; }
    touch "$FKB/.shared_applied"
  else echo "[2] .shared_applied exists, skip"; fi
  # keepmeta is shared with soft2 and already exists; regenerate only if missing
  if [ ! -f "$KEEPMETA" ]; then
    echo "[3 $(ts)] redist_prep_v7coder -> $KEEPMETA"
    "$PY" "$SCR/redist_prep_v7coder.py" "$FKB" "$DROP" "$KEEPMETA" \
      || { echo "FATAL keepmeta prep"; exit 5; }
  else echo "[3] keepmeta exists, skip"; fi
else echo "[1-3] COMBO exists, skip student build"; fi

# ── 4. DERN fold mergemoe (single-variable swap) ─────────
if [ ! -f "$COMBO/model.safetensors.index.json" ] && [ ! -f "$COMBO/model.safetensors" ]; then
  echo "[4 $(ts)] redist_dern_eq11_fold --fold-mode mergemoe --assign-topk 2 -> $COMBO (GPU$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$DERN" --teacher "$SRC" --student "$FKB" --keep-meta "$KEEPMETA" \
      --freq-corpus "$CORPUS" --seq-corpus "$CORPUS" --out "$COMBO" --device cuda:0 \
      --assign-topk 2 --fold-mode mergemoe || { echo "FATAL redist fold"; exit 7; }
  echo "[4 $(ts)] DERN-fold done; removing intermediate student $FKB"
  rm -rf "$FKB"
else echo "[4] $COMBO exists, skip redist"; fi

# ── 5. convert F16 ───────────────────────────────────────
[ -f "$F16" ] || { echo "[5 $(ts)] convert -> $F16";
  "$PY" "$LCPP/convert_hf_to_gguf.py" "$COMBO" --outfile "$F16" --outtype f16 \
    || { echo "FATAL convert"; exit 8; }; }

# ── 6. model-specific imatrix (calib_both, 128 chunks, ngl 99) — PRESERVED ──
if [ ! -f "$IMAT" ]; then
  echo "[6 $(ts)] llama-imatrix -ngl 99 --chunks 128 (GPU$GPU) -> $IMAT"
  CUDA_VISIBLE_DEVICES=$GPU "$LCPP/build/bin/llama-imatrix" \
    -m "$F16" -f "$CALIB" -o "$IMAT" --chunks 128 -ngl 99 \
    > "$SFT/fkbroad_lsfold_imatrix_build.log" 2>&1 \
    || { echo "FATAL imatrix"; tail -25 "$SFT/fkbroad_lsfold_imatrix_build.log"; exit 9; }
fi
echo "[6 $(ts)] imatrix.dat $(stat -c%s "$IMAT" 2>/dev/null) bytes (PRESERVED)"

# ── 7. imat-Q6_K + no-imat Q4_0, then drop F16 ───────────
[ -f "$Q6" ] || { echo "[7a $(ts)] quantize imat-Q6_K -> $Q6";
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 \
    || { echo "FATAL quant Q6"; exit 10; }; }
[ -f "$Q40" ] || { echo "[7b $(ts)] quantize no-imat Q4_0 -> $Q40 (matches reg_q40 control build)";
  "$LCPP/build/bin/llama-quantize" "$F16" "$Q40" Q4_0 32 \
    || { echo "FATAL quant Q4_0"; exit 10; }; }
for g in "$Q6" "$Q40"; do
  magic=$("$PY" -c "import sys;print(open('$g','rb').read(4).decode('latin1'))" 2>/dev/null)
  [ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF header $g"; exit 10; }
done
echo "[7 $(ts)] Q6 $(du -h "$Q6"|cut -f1)  Q4_0 $(du -h "$Q40"|cut -f1) ; dropping F16"
rm -f "$F16"

# ── 8. loop gate imat-Q6 {0.9,0.8}x48 (DECISIVE — must stay 0/48 like v8) ──
echo "[8 $(ts)] loop gate lsfold imat-Q6 GPU$GPU:8260 vs v8-avg imat-Q6 (0/48,0/48)"
bash "$GATE" "$Q6" "$GPU" 8260 "$GRES_Q6" v8-lsfold-imatQ6 || echo "[8] WARN gate rc=$?"

# ── 9. loop gate no-imat Q4_0 {0.9,0.8}x48 (vs v8-avg 9/48,12/48) ─────────
echo "[9 $(ts)] loop gate lsfold no-imat-Q4_0 GPU$GPU:8261 vs v8-avg no-imat-Q4_0 (9/48,12/48)"
bash "$GATE" "$Q40" "$GPU" 8261 "$GRES_Q40" v8-lsfold-noimatQ40 || echo "[9] WARN gate rc=$?"

# ── 10. summary ──────────────────────────────────────────
echo "[10 $(ts)] === LS-FOLD BUILD + LOOP GATES DONE ==="
for R in "$GRES_Q6" "$GRES_Q40"; do
  echo "loop gate: $R"
  [ -f "$R" ] && "$PY" -c "import json;d=json.load(open('$R'));[print(' ',r['config'],'loops',r['loops'],'/',r['seeds']) for r in d['results']]" 2>/dev/null
done
echo "imatrix preserved: $IMAT"
echo "lsfold bf16: $COMBO ; imat-Q6: $Q6 ; no-imat-Q4_0: $Q40"
echo "compare vs v8-average: imat-Q6 0/48,0/48 | no-imat-Q4_0 9/48,12/48"
echo "###### LSFOLD_BUILD_DONE $(ts) ######"
