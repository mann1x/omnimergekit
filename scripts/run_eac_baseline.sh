#!/bin/bash
# run_eac_baseline.sh — PREPARED runner for T18 Step-2 EAC-MoE TopK-MSE router
# recalibration on top of a 62e v1-coder variant. PREPARE/REVIEW tool: it
# DEFAULTS TO DRY-RUN (prints the exact plan, touches nothing). Pass --run to
# actually execute. Built for council review (T137); see EAC_COUNCIL_BRIEF.md.
#
# What EAC does: pairs base-128e (teacher) and the pruned variant (student) on
# the SAME calibration tokens, then optimizes ONLY the variant's
# router.proj.weight to match the teacher's top-K routing logits (Eq.5). It
# edits the variant IN PLACE (shards backed up to *.pre_eac_calibrate), so this
# runner first COPIES the chosen variant to <variant>-eac-it and calibrates the
# copy — the original (e.g. the published baseline) is never touched.
#
# Corpus: WikiText-2 + router_calib_corpus.txt, built by eac_prepare_corpus.py as
# a balanced 50/50 256k window (n_seq=128 x seq_len=2048, 64 calib + 64 wiki
# sequences, consumed whole — the chosen tradeoff vs the ~tens-of-hours full
# corpus). n_seq is read from the corpus meta. seq-len 2048, steps 120, calib-k
# 16 (EAC window, wider than Gemma native top-8).
#
# Usage:
#   run_eac_baseline.sh                       # dry-run plan for the baseline
#   run_eac_baseline.sh --target E1           # dry-run for E1heplus
#   run_eac_baseline.sh --target baseline --run          # actually calibrate
#   run_eac_baseline.sh --target baseline --run --with-eval  # + Q6_K + HE+/MPE
set -uo pipefail
BM=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
PY=/root/anaconda3/envs/omnimergekit/bin/python
EAC=$BM/scripts/router_eac_calibrate.py
PREP=$BM/scripts/eac_prepare_corpus.py
QG=$BM/scripts/quantize_gguf.py
OMK=/shared/dev/omnimergekit/eval/omk_eval.py
BASE128=$BM/google/gemma-4-26B-A4B-it
CORPUS=$BM/scripts/eac_corpus_wiki2_plus_calib.txt
CAL=$BM/scripts/calibration_datav5.txt

# ---- params (user spec) ----
SEQ_LEN=2048
STEPS=120
CALIB_K=16
MAX_GPU_GIB=20
MAX_CPU_GIB=400

TARGET=baseline
DO_RUN=0
WITH_EVAL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET=$2; shift 2;;
    --run) DO_RUN=1; shift;;
    --with-eval) WITH_EVAL=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

case "$TARGET" in
  baseline) TAG=62e-fc15_25-p8; MAP=$BM/scripts/v6coder_C6v3lcb_62e_fc15_25_p8_drop_map.json;;
  E1)       TAG=62e-E1heplus;   MAP=$BM/scripts/v6coder_C6v3lcb_62e_E1_heplus_drop_map.json;;
  E2)       TAG=62e-E2lcbhep;   MAP=$BM/scripts/v6coder_C6v3lcb_62e_E2_lcbhep_drop_map.json;;
  *) echo "FATAL: --target must be baseline|E1|E2"; exit 2;;
esac
VDIR_SRC=$BM/google/gemma-4-A4B-${TAG}-it
VDIR_EAC=$BM/google/gemma-4-A4B-${TAG}-eac-it
CACHE=$BM/eac_cache_${TAG}

# n_seq comes from the corpus meta (cover-all). Compute on the fly if missing.
META=$CORPUS.meta.json
if [ -f "$META" ]; then
  N_SEQ=$("$PY" -c "import json;print(json.load(open('$META'))['n_seq'])")
else
  N_SEQ="<run eac_prepare_corpus.py first>"
fi

EAC_CMD=("$PY" "$EAC" --phase both
  --base-dir "$BASE128" --variant-dir "$VDIR_EAC" --drop-map "$MAP"
  --corpus-file "$CORPUS" --cache-dir "$CACHE"
  --n-seq "$N_SEQ" --seq-len "$SEQ_LEN" --steps "$STEPS" --calib-k "$CALIB_K"
  --max-gpu-gib "$MAX_GPU_GIB" --max-cpu-gib "$MAX_CPU_GIB")

echo "=================== EAC PLAN (target=$TARGET) ==================="
echo "  base (teacher) : $BASE128"
echo "  variant src    : $VDIR_SRC   (preserved)"
echo "  variant copy   : $VDIR_EAC   (calibrated in place)"
echo "  drop-map       : $MAP"
echo "  corpus         : $CORPUS  (WikiText-2 + router_calib_corpus.txt, balanced 50/50 256k window = 128x2048)"
echo "  meta           : $META"
echo "  n_seq=$N_SEQ  seq_len=$SEQ_LEN  steps=$STEPS  calib_k=$CALIB_K"
echo "  cache-dir      : $CACHE   (persistent disk, NEVER /tmp)"
echo "  with-eval      : $WITH_EVAL   (Q6_K + HE+164 + MPE-300 vs pre-EAC)"
echo "  EAC command:"
printf '    %q ' "${EAC_CMD[@]}"; echo
echo "================================================================"

if [ "$DO_RUN" -ne 1 ]; then
  echo "[dry-run] nothing executed. Re-run with --run to calibrate."
  exit 0
fi

# ---------- live execution (only with --run) ----------
[ -f "$CORPUS" ] || { echo "FATAL: corpus missing — run: $PY $PREP --seq-len $SEQ_LEN"; exit 1; }
[ "$N_SEQ" -ge 1 ] 2>/dev/null || { echo "FATAL: n_seq not resolved from meta"; exit 1; }
[ -f "$VDIR_SRC/model.safetensors.index.json" ] || { echo "FATAL: variant missing $VDIR_SRC"; exit 1; }
# GPU-busy guard: refuse to collide with another job on the targeted GPU.
# Honors CUDA_VISIBLE_DEVICES — when set, only checks that specific GPU index
# so parallel GPU0/GPU1 dispatching (run_shared_x_pes_ablation on GPU0 + EAC
# on GPU1) works. Without CVD, checks all GPUs.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  busy_count=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  guard_scope="GPU $CUDA_VISIBLE_DEVICES"
else
  busy_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  guard_scope="any GPU"
fi
if [ "$busy_count" -gt 0 ]; then
  echo "FATAL: a GPU compute process is already running on $guard_scope — refusing to launch EAC."
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  else
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  fi
  exit 1
fi
TS=$(date +%Y%m%d_%H%M%S)
LOG=$BM/logs/eac_${TAG}_${TS}.log
mkdir -p "$BM/logs" "$CACHE"
echo "[eac] copying $VDIR_SRC -> $VDIR_EAC (preserve original)" | tee -a "$LOG"
# Flush pending writes before reading — the source variant may have been
# rewritten by router_shared_upweight.py / expert_drop.py moments ago, and
# rsync occasionally hits ENODATA on writeback-pending ext4 pages.
sync
# Retry-on-failure: rsync once, verify all shards match in size, retry-deep on miss.
copy_attempt=0
while :; do
  copy_attempt=$((copy_attempt+1))
  rsync -a --delete "$VDIR_SRC/" "$VDIR_EAC/" 2>&1 | tee -a "$LOG"
  rsync_rc=${PIPESTATUS[0]}
  # post-copy verification: every safetensors shard must have matching size + readable header
  bad=0
  for src in "$VDIR_SRC"/model-*.safetensors; do
    fn=$(basename "$src"); dst="$VDIR_EAC/$fn"
    if [ ! -f "$dst" ] || [ "$(stat -c%s "$src")" != "$(stat -c%s "$dst")" ]; then
      echo "[eac] verify: shard $fn size mismatch (src=$(stat -c%s "$src") dst=$(stat -c%s "$dst" 2>/dev/null || echo MISSING))" | tee -a "$LOG"
      bad=$((bad+1))
    fi
  done
  if [ "$rsync_rc" -eq 0 ] && [ "$bad" -eq 0 ]; then
    echo "[eac] copy verified ($copy_attempt attempt$([ $copy_attempt -gt 1 ] && echo s))" | tee -a "$LOG"
    break
  fi
  if [ "$copy_attempt" -ge 3 ]; then
    echo "FATAL: rsync failed 3× (rc=$rsync_rc bad_shards=$bad) — disk or filesystem issue" | tee -a "$LOG"
    exit 1
  fi
  echo "[eac] copy attempt $copy_attempt failed (rc=$rsync_rc bad_shards=$bad) — flushing + retrying" | tee -a "$LOG"
  sync; sleep 5
done
echo "[eac] launching calibration ($(date -Iseconds))" | tee -a "$LOG"
"${EAC_CMD[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[eac] calibration exit=$rc" | tee -a "$LOG"
[ "$rc" -eq 0 ] || exit "$rc"

if [ "$WITH_EVAL" -eq 1 ]; then
  export PATH=/root/anaconda3/envs/omnimergekit/bin:/opt/llama.cpp/build/bin:$PATH
  export HF_ALLOW_CODE_EVAL=1 OMK_NO_README=1
  OUTG=$VDIR_EAC-GGUF
  echo "[eac] quantize Q6_K -> $OUTG" | tee -a "$LOG"
  "$PY" "$QG" --model "$VDIR_EAC" --output-dir "$OUTG" --only Q6_K \
      --base-model-id google/gemma-4-26B-A4B-it --cal-data "$CAL" \
      --no-upload --keep-local --sanity-check 2>&1 | tee -a "$LOG"
  Q6=$OUTG/gemma-4-A4B-${TAG}-eac-it-Q6_K.gguf
  RES=$BM/eval_results_t137_eac
  # 21q rumination screen FIRST (mandatory gate per T137 council 2026-05-27 verdict).
  # The screen is 3 HE+ + 3 IFEval + 15 MPE problems hand-selected as the
  # rumination-only subset (RUNAWAY/LOOPING/RUMINATE-FAIL) for fc15_25-p8.
  # Pure HE+164 + MPE-300 are capability-blind to loops (R1 cured loops but
  # cratered Rust −32 → seemed fine on the screen alone). Always dual-axis.
  for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15 humanevalplus_full multipl_e_100; do
    "$PY" "$OMK" --model "$Q6" --tokenizer "$VDIR_EAC" --template "$tpl" --backend llama \
        --served-name "t137-${TAG}-eac" --results-dir "$RES" 2>&1 | tee -a "$LOG"
  done
fi
echo "[eac] DONE ($(date -Iseconds))" | tee -a "$LOG"
