#!/usr/bin/env bash
# run_agentic_diff.sh -- broadened agentic clean-vs-loop router DIFFERENTIAL.
#
# "differential + additive" step 1 (the differential). Reconstructs the fkbroad
# SELECTION bf16 (no-fold student = expert_drop(fkbroad) + shared a=1.2; the A2
# looper is gone, no loop text was saved, so we must regenerate) as the HF looper,
# then runs router_diff_agentic.py on GPU0:
#   fkbroad generates agentic loops -> 128e teacher-forces -> per-(layer,expert)
#   DROPPED mass on loop vs clean tokens, ranked. Decides if force-keep is viable.
# Reconstruct is CPU/disk only (no GPU) so it never contends with the no-fold loop
# gate on GPU0. PID-kill only.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
DROP=/srv/ml/scripts/v8coder_fkbroad_drop_map.json
DIFF=/srv/ml/scripts/router_diff_agentic.py
LOOPER=/mnt/sdc/ml/sft_heal/fkbroad-selection-looper-it
OUT=/mnt/sdc/ml/google/expert_neuron_v8_agentic_diff.json
GPU=0
ts(){ date '+%T %Z'; }
echo "==================== agentic router differential $(ts) ===================="

# ── preflight ─────────────────────────────────────────────
for f in "$SRC/config.json" "$DROP" "$DIFF" "$SCR/expert_drop.py" \
         "$RECIPE/router_shared_upweight.py"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc '0-9')
echo "[preflight $(ts)] ${free}G free on /mnt/sdc"
# build headroom (~80G for fp32 looper) is ONLY needed when the looper must be built.
# If it already exists, the differential writes only a ~200MB JSON.
if [ ! -f "$LOOPER/.shared_applied" ]; then
  [ "${free:-0}" -lt 80 ] && { echo "FATAL <80G free — looper fp32 build needs it; reclaim first"; exit 9; }
else
  echo "[preflight $(ts)] looper exists; skipping build-headroom check (run writes ~200MB)"
  [ "${free:-0}" -lt 3 ] && { echo "FATAL <3G free — refusing even the output write"; exit 9; }
fi

# ── 1. reconstruct fkbroad SELECTION looper (CPU/disk, no GPU) ──
if [ ! -f "$LOOPER/model.safetensors.index.json" ] && [ ! -f "$LOOPER/model.safetensors" ]; then
  echo "[1 $(ts)] expert_drop(fkbroad) -> $LOOPER"
  "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$LOOPER" \
    || { echo "FATAL expert_drop"; exit 3; }
  [ -f "$LOOPER/tokenizer.json" ] || { echo "FATAL tokenizer.json not copied"; exit 3; }
else echo "[1] $LOOPER exists, skip"; fi
if [ ! -f "$LOOPER/.shared_applied" ]; then
  echo "[2 $(ts)] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
  "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$LOOPER" \
    --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared_upweight"; exit 4; }
  touch "$LOOPER/.shared_applied"
else echo "[2] .shared_applied exists, skip"; fi

# ── 3. differential on GPU1 ──────────────────────────────
echo "[3 $(ts)] router_diff_agentic GPU$GPU (looper Phase1 -> 128e Phase2)"
# T223-reopen: FULL UNTRUNCATED fixtures (all 9, incl 87k csharp). Eager attn + chunked
# prefill (chunk=2048) bounds per-chunk attention to [heads,2048,cache_len]; validated
# statistic-neutral on the 5 global hd512 layers (probe_chunked_equiv: pearson=1.0, top20 20/20).
# Big fixtures FIRST so any Phase-1 OOM surfaces immediately after the looper build.
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" "$DIFF" --looper "$LOOPER" --drop-map "$DROP" --out "$OUT" \
    --fixtures csharp_loop_parity,csharp_loop_parity_nothink,solar_system_v4,kvstore_deep,solar_build_friction,solar_build_start,agentic_filewrite_v1,agentic_plan_todo_v1,threejs_build \
    --seeds 6 --gen-tokens 2048 --prefill-chunk 2048 --teacher-chunk 1024 \
  || { echo "FATAL differential"; exit 5; }

echo "[4 $(ts)] === AGENTIC DIFFERENTIAL DONE -> $OUT ==="
echo "###### AGENTIC_DIFF_DONE $(ts) ######"
