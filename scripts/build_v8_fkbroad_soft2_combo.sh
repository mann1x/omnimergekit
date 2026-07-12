#!/usr/bin/env bash
# build_v8_fkbroad_soft2_combo.sh — T212 v8-coder combo: force-keep map (fkbroad) + DERN soft-top-2.
#
# APPLES-TO-APPLES vs soft2 (dern11-soft-soft2). soft2 = C6v3lcb selection + DERN Eq.11 --assign-topk 2
# + shared a=1.2, then model-specific imat-Q6 (0/48 loops @ vendor_minp_rep t0.9). This combo changes
# the ONLY thing we want to test: the SELECTION. v8coder_fkbroad force-keeps the 30 broad code/LCB
# experts the C6v3lcb prune dropped (evicting 30 low-agg multilingual/creative survivors), so the
# broad code/LCB capacity survives DIRECTLY instead of via the lossy DERN fold. Everything else —
# shared a=1.2, DERN Eq.11 soft-top-2, eog_corpus_solar, calib_both imatrix, imat-Q6 — is identical
# to soft2. NO eog force-keep (that HURT loops in T206). Pipeline:
#   expert_drop(fkbroad) -> router_shared_upweight(a=1.2) -> redist_prep -> redist_dern_eq11(--assign-topk 2)
#   -> F16 -> model-specific imatrix(calib_both,128,ngl99) -> imat-Q6 -> loop gate {0.9,0.8}x48
#   -> trade-check HE+164 / MPE-100 / LCB-55 (greedy) vs soft2.
# Self-scheduling: CPU pre-stages run now; GPU stages wait for the first GPU <2000 MiB. Resumable.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
DERN=/srv/ml/repos/omnimergekit/scripts/redist_dern_eq11.py
LCPP=/mnt/sdc/ml/llama.cpp-latest
SFT=/mnt/sdc/ml/sft_heal
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it          # 128e teacher + tokenizer
DROP=/srv/ml/scripts/v8coder_fkbroad_drop_map.json  # the force-keep map (T212)
CORPUS=$SFT/eog_corpus_solar.jsonl                  # DERN fold corpus (same as soft2)
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt       # imatrix calib (same as soft2)
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
AL=/srv/ml/agentic_loop

FKB=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-it        # student bf16 (expert_drop + shared)
KEEPMETA=$SFT/v7coder_fkbroad_keepmeta.json
COMBO=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-it # combo bf16 (after DERN soft-top-2)
F16=$SFT/fkbroad-soft2-F16.gguf
IMAT=$SFT/fkbroad-soft2-imatrix.dat                 # PRESERVED (mandatory)
Q6=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-imat-Q6_K.gguf
GRES=$AL/results/fkbroad-soft2-imatQ6_minp48.json
TD=$AL/results/fkbroad_soft2_tradecheck
NAME=fkbroad-soft2-imatq6
PORT=8190
ts(){ date '+%T %Z'; }
echo "==================== build v8 fkbroad+soft2 combo $(ts) ===================="

# ── preflight ─────────────────────────────────────────────
for f in "$SRC/config.json" "$DROP" "$CORPUS" "$CALIB" "$SCR/expert_drop.py" \
         "$RECIPE/router_shared_upweight.py" "$SCR/redist_prep_v7coder.py" "$DERN" \
         "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-imatrix" \
         "$LCPP/build/bin/llama-quantize" "$GATE" "$OMK"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
grep -q -- "--assign-topk" "$DERN" || { echo "FATAL redist not patched (no --assign-topk)"; exit 2; }
echo "[preflight $(ts)] disk:"; df -h "$SFT" | tail -1

# ── stages 1-3 build the fkbroad student bf16 (skip entirely if COMBO already folded) ──
if [ ! -f "$COMBO/model.safetensors.index.json" ] && [ ! -f "$COMBO/model.safetensors" ]; then
  # 1. expert_drop (fkbroad keep set) — CPU, runs now
  if [ ! -f "$FKB/model.safetensors.index.json" ] && [ ! -f "$FKB/model.safetensors" ]; then
    echo "[1 $(ts)] expert_drop(fkbroad) -> $FKB"
    "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$FKB" \
      || { echo "FATAL expert_drop"; exit 3; }
    [ -f "$FKB/tokenizer.json" ] || { echo "FATAL tokenizer.json not copied"; exit 3; }
  else echo "[1] $FKB exists, skip"; fi

  # 2. router_shared_upweight (a=1.2, bf16) — CPU
  if [ ! -f "$FKB/.shared_applied" ]; then
    echo "[2 $(ts)] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
    "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$FKB" \
      --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared_upweight"; exit 4; }
    touch "$FKB/.shared_applied"
  else echo "[2] .shared_applied exists, skip"; fi

  # 3. keep-meta from fkbroad drop map — CPU
  if [ ! -f "$KEEPMETA" ]; then
    echo "[3 $(ts)] redist_prep_v7coder -> $KEEPMETA"
    "$PY" "$SCR/redist_prep_v7coder.py" "$FKB" "$DROP" "$KEEPMETA" \
      || { echo "FATAL keepmeta prep"; exit 5; }
  fi
else echo "[1-3] COMBO exists, skip student build"; fi

# ── acquire first free GPU (<2000 MiB), up to 4h ─────────
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

# ── 4. DERN Eq.11 soft-top-2 fold ────────────────────────
if [ ! -f "$COMBO/model.safetensors.index.json" ] && [ ! -f "$COMBO/model.safetensors" ]; then
  echo "[4 $(ts)] redist_dern_eq11 --assign-topk 2 -> $COMBO (GPU$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$DERN" --teacher "$SRC" --student "$FKB" --keep-meta "$KEEPMETA" \
      --freq-corpus "$CORPUS" --seq-corpus "$CORPUS" --out "$COMBO" --device cuda:0 \
      --assign-topk 2 || { echo "FATAL redist_dern_eq11"; exit 7; }
  # free the fkbroad student bf16 (reproducible from map; tight disk)
  echo "[4 $(ts)] DERN done; removing intermediate student $FKB"
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
    > "$SFT/fkbroad_soft2_imatrix_build.log" 2>&1 \
    || { echo "FATAL imatrix"; tail -25 "$SFT/fkbroad_soft2_imatrix_build.log"; exit 9; }
fi
echo "[6 $(ts)] imatrix.dat $(stat -c%s "$IMAT" 2>/dev/null) bytes"

# ── 7. imat-Q6_K, then drop F16 ──────────────────────────
[ -f "$Q6" ] || { echo "[7 $(ts)] quantize imat-Q6_K -> $Q6";
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 \
    || { echo "FATAL quant"; exit 10; }; }
magic=$("$PY" -c "import sys;print(open('$Q6','rb').read(4).decode('latin1'))" 2>/dev/null)
[ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF header"; exit 10; }
rm -f "$F16"

# ── 8. loop gate vendor_minp_rep {0.9, 0.8} x 48 ─────────
echo "[8 $(ts)] loop gate $NAME {t0.9,t0.8} GPU$GPU:$PORT vs soft2 (0/48,0/48)"
bash "$GATE" "$Q6" "$GPU" "$PORT" "$GRES" "$NAME" || echo "[8] WARN gate rc=$?"

# ── 9. trade-check HE+164 / MPE-100 / LCB-55 (greedy) ────
mkdir -p "$TD"
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export HF_ALLOW_CODE_EVAL=1
p=8201
for TPL in humanevalplus_full multipl_e_100 lcb_medium_55_v4; do
  echo "[9 $(ts)] trade-check $TPL (GPU$GPU:$p)"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$OMK" \
    --model "$Q6" --template "$TPL" --backend llama --quant gguf \
    --port "$p" --results-dir "$TD" --served-name "$NAME" \
    --tokenizer "$SRC" --parallel 2 || echo "[9] WARN $TPL rc=$?"
  p=$((p+1))
done

# ── 10. summary ──────────────────────────────────────────
echo "[10 $(ts)] === FKBROAD-SOFT2 COMBO DONE ==="
echo "loop gate: $GRES"; [ -f "$GRES" ] && "$PY" -c "import json;d=json.load(open('$GRES'));[print(r['config'],'loops',r['loops'],'/',r['seeds']) for r in d['results']]" 2>/dev/null
for TPL in humanevalplus_full multipl_e_100 lcb_medium_55_v4; do
  S="$TD/$TPL/$NAME/summary.json"
  [ -f "$S" ] && "$PY" -c "import json;d=json.load(open('$S'));print('$TPL','score',d.get('score'),d.get('scores'))" 2>/dev/null
done
echo "imatrix preserved: $IMAT"
echo "combo bf16: $COMBO ; imat-Q6: $Q6"
echo "compare vs soft2: loops 0/48,0/48 | HE+ MPE LCB-55 (see soft2_imat_*.log)"
