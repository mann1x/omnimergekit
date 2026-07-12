#!/bin/bash
# run_shared_x_pes_ablation.sh — T141 2×2 ablation of (shared_alpha × per_expert_scale)
# on the 62e-fc15_25-p8 baseline. DRY-RUN default; pass --run to execute.
#
# Why this matters:
#   The current published baseline (fc15_25-p8-it) has shared α=1.2 baked in
#   (via router_shared_upweight.py on mlp.down_proj.weight). T139 found per-expert
#   α=1.20 also caused HE+164 to fully recover (zero capability cost). Both knobs
#   are +20% magnitude on different paths (always-on shared FFN vs routed-mixture
#   gain). The ablation table isolates which contribution is doing the work and
#   whether they compose (super-additive, additive, or interfering).
#
# Cells (sX=shared α, pY=per-expert α):
#   A1 (s=1.0, p=1.0)  — pure C6 floor; ZERO upweight (NEW bf16; invert shared via ×1/1.2)
#   A2 (s=1.0, p=1.20) — PES alone     (NEW bf16; invert shared, then PES)
#   B1 (s=1.2, p=1.0)  — current published baseline (REUSE fc15_25-p8-it as-is)
#   B2 (s=1.2, p=1.20) — both knobs    (REUSE fc15_25-p8-pes1_20-it from T139)
#
# Eval stack pinning:
#   Per feedback_pin_eval_stack_across_cohort, all 4 cells get RE-EVAL on whatever
#   pod/host runs this script — solidpc evals stay as historical reference but
#   the published 2×2 must be single-stack. Default execution is local; pass
#   --pod NAME (SSH alias) to delegate to a remote host (must have omk env +
#   llama.cpp built, e.g. linode-blackswan-2 after pod_eval_bootstrap.sh).
#
# Sequence per cell (build phase):
#   1. cp -al base→cell dir (hardlink-copy, fast, 0 disk IO for unchanged shards)
#   2. invert shared (×0.83333) if s=1.0
#   3. apply per_expert_rescale α=1.20 if p=1.20
#   4. quantize_gguf.py --only Q6_K --cal-data calv5 (per-cell imatrix)
# Sequence per cell (eval phase):
#   5. omk_eval humanevalplus_full (HE+164)
#   6. omk_eval multipl_e_100      (MPE-300, 100×{rust,java,js})
#
# Total wall-clock estimate (one cell at a time, single Blackwell sm_120 GPU):
#   - Build: A1 + A2 ≈ 35 min combined (bf16 ops are CPU; imatrix is ~10 min/cell)
#   - Quants: 4× Q6_K ≈ 30 min combined (Blackwell ≈ 2× faster than 3090)
#   - Evals: 4 × (HE+164 ~15min + MPE-300 ~20min) ≈ 2.5 hours
#   - Grand total: ~3.5 hours sequential
#   - With --parallel-gpus 0,1: ~1.75 hours (two cells run concurrently)
#
# Usage:
#   run_shared_x_pes_ablation.sh                  # dry-run plan, no compute
#   run_shared_x_pes_ablation.sh --run            # execute locally
#   run_shared_x_pes_ablation.sh --run --pod linode-blackswan-2
#   run_shared_x_pes_ablation.sh --cells "A1 A2" --run  # subset
#   run_shared_x_pes_ablation.sh --run --parallel-gpus 0,1  # split cells across GPUs
#
# Outputs:
#   - bf16 dirs: google/gemma-4-A4B-62e-fc15_25-p8-s{1_0|1_2}p{1_0|1_20}-it/
#   - GGUFs:     <bf16>-GGUF/ with F16.gguf, imatrix.dat, Q6_K.gguf
#   - evals:     eval_results_t141_shared_x_pes/<template>/t141-62e-<cell>/summary.json
#   - markers:   logs/t141_<cell>_DONE
#   - 2×2 table printed at end (read summary.json .score, NEVER raw exact_match)

set -uo pipefail

# ============================ paths / constants ============================
BM=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
PY=/root/anaconda3/envs/omnimergekit/bin/python
SH_UP=$BM/scripts/router_shared_upweight.py
PES=$BM/scripts/router_per_expert_rescale.py
QG=$BM/scripts/quantize_gguf.py
OMK=/shared/dev/omnimergekit/eval/omk_eval.py
CAL=$BM/scripts/calibration_datav5.txt

# baselines we reuse without modification
BASE_S12_P10=$BM/google/gemma-4-A4B-62e-fc15_25-p8-it          # B1: s=1.2, p=1.0
BASE_S12_P120=$BM/google/gemma-4-A4B-62e-fc15_25-p8-pes1_20-it # B2: s=1.2, p=1.20

# ============================ args ============================
DO_RUN=0
POD=""
CELLS="A1 A2 B1 B2"
PARALLEL_GPUS=""
SKIP_EXISTING=1
EVALS="humanevalplus_full multipl_e_100"
while [ $# -gt 0 ]; do
  case "$1" in
    --run)             DO_RUN=1; shift;;
    --pod)             POD=$2; shift 2;;
    --cells)           CELLS=$2; shift 2;;
    --parallel-gpus)   PARALLEL_GPUS=$2; shift 2;;
    --no-skip-existing) SKIP_EXISTING=0; shift;;
    --evals)           EVALS=$2; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

# ============================ cell descriptor table ============================
# format: cell_id | shared_alpha | pes_alpha | source_action
# A1 (s=1.0, p=1.0):  copy fc15_25-p8-it → ablation dir, invert shared (×1/1.2)
# A2 (s=1.0, p=1.20): copy fc15_25-p8-it → ablation dir, invert shared, then PES α=1.20
# B1 (s=1.2, p=1.0):  reuse existing fc15_25-p8-it (no action)
# B2 (s=1.2, p=1.20): reuse existing fc15_25-p8-pes1_20-it (no action)
declare -A SHARED PEXP DIR
SHARED[A1]=1.0; PEXP[A1]=1.0;  DIR[A1]=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_0-it
SHARED[A2]=1.0; PEXP[A2]=1.20; DIR[A2]=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it
SHARED[B1]=1.2; PEXP[B1]=1.0;  DIR[B1]=$BASE_S12_P10
SHARED[B2]=1.2; PEXP[B2]=1.20; DIR[B2]=$BASE_S12_P120

build_cell() {
  local cell=$1
  local out=${DIR[$cell]}
  local s=${SHARED[$cell]}
  local p=${PEXP[$cell]}
  local log=$BM/logs/t141_${cell}.log
  mkdir -p "$BM/logs"
  echo "[$cell s=$s p=$p] -> $out" | tee -a "$log"

  case $cell in
    B1|B2)
      if [ -f "$out/model.safetensors.index.json" ]; then
        echo "  reuse existing bf16 (no build needed)" | tee -a "$log"
      else
        echo "FAIL [$cell]: expected pre-built bf16 missing: $out" | tee -a "$log"; return 1
      fi
      ;;
    A1)
      if [ -f "$out/model.safetensors.index.json" ] && [ $SKIP_EXISTING -eq 1 ]; then
        echo "  skip-existing: bf16 exists" | tee -a "$log"
      else
        echo "  cp -al $BASE_S12_P10 -> $out" | tee -a "$log"
        cp -al "$BASE_S12_P10" "$out" 2>&1 | tee -a "$log" || return 1
        # invert shared α=1.2 -> α=1.0 via ×1/1.2
        "$PY" "$SH_UP" --model-dir "$out" --target mlp.down_proj.weight \
            --alpha 0.8333333333333333 2>&1 | tee -a "$log" || return 1
        # marker noting state
        echo "shared_alpha=1.0 (inverted from baseline 1.2 via x0.8333)" > "$out/.shared_state"
        rm -f "$out/.shared_applied"
      fi
      ;;
    A2)
      if [ -f "$out/model.safetensors.index.json" ] && [ $SKIP_EXISTING -eq 1 ]; then
        echo "  skip-existing: bf16 exists" | tee -a "$log"
      else
        echo "  cp -al $BASE_S12_P10 -> $out" | tee -a "$log"
        cp -al "$BASE_S12_P10" "$out" 2>&1 | tee -a "$log" || return 1
        # invert shared first
        "$PY" "$SH_UP" --model-dir "$out" --target mlp.down_proj.weight \
            --alpha 0.8333333333333333 2>&1 | tee -a "$log" || return 1
        echo "shared_alpha=1.0 (inverted)" > "$out/.shared_state"
        rm -f "$out/.shared_applied"
        # PES rescale in-place (omitting --out-dir edits in place with .pre_per_expert_rescale backup)
        "$PY" "$PES" --model-dir "$out" --alpha 1.20 2>&1 | tee -a "$log" || return 1
        echo "pes_alpha=1.20" > "$out/.pes_applied"
      fi
      ;;
  esac

  # ---- Q6_K quant (per-cell imatrix from cal-data) ----
  local outg=$out-GGUF
  local q6=$outg/$(basename "$out")-Q6_K.gguf
  if [ -f "$q6" ] && [ $SKIP_EXISTING -eq 1 ]; then
    echo "  Q6_K exists -> skip quant" | tee -a "$log"
  else
    mkdir -p "$outg"
    "$PY" "$QG" --model "$out" --output-dir "$outg" --only Q6_K \
        --base-model-id google/gemma-4-26B-A4B-it --cal-data "$CAL" \
        --no-upload --keep-local --sanity-check 2>&1 | tee -a "$log" \
      || { echo "FAIL quant [$cell]" | tee -a "$log"; return 1; }
  fi

  # ---- Evals ----
  local res=$BM/eval_results_t141_shared_x_pes
  for tpl in $EVALS; do
    "$PY" "$OMK" --model "$q6" --tokenizer "$out" --template "$tpl" --backend llama \
        --served-name "t141-62e-${cell}" --results-dir "$res" 2>&1 | tee -a "$log" \
      || echo "  WARN omk_eval ${cell}/${tpl} nonzero" | tee -a "$log"
  done
  touch "$BM/logs/t141_${cell}_DONE"
}

# ============================ plan ============================
echo "=================== T141: shared×PES 2×2 ABLATION PLAN ==================="
echo "  baseline (B1, reuse) : $BASE_S12_P10"
echo "  baseline (B2, reuse) : $BASE_S12_P120"
echo "  cells to run         : $CELLS"
echo "  evals per cell       : $EVALS"
echo "  cal-data             : $CAL"
echo "  remote pod           : ${POD:-<local>}"
echo "  parallel GPUs        : ${PARALLEL_GPUS:-<sequential>}"
echo "  skip-existing        : $SKIP_EXISTING"
echo "  with-run             : $DO_RUN"
echo
echo "  | cell |  s  |   p   | bf16 dir                                                |"
echo "  |------|-----|-------|---------------------------------------------------------|"
for c in $CELLS; do
  printf "  | %-2s   | %-3s | %-5s | %s\n" "$c" "${SHARED[$c]}" "${PEXP[$c]}" "${DIR[$c]}"
done
echo "=========================================================================="

if [ "$DO_RUN" -ne 1 ]; then
  echo "[dry-run] nothing executed. Re-run with --run."
  exit 0
fi

# ============================ live execution ============================
# Remote-execution mode: rsync this script to pod, run with same args minus --pod
if [ -n "$POD" ]; then
  echo "[t141] delegating to remote pod $POD" | tee -a "$BM/logs/t141_dispatch.log"
  rsync -ah --progress "$BM/scripts/run_shared_x_pes_ablation.sh" "$POD:/workspace/backup_models/scripts/"
  ssh -t "$POD" "cd /workspace/backup_models && bash scripts/run_shared_x_pes_ablation.sh --run --cells \"$CELLS\" --evals \"$EVALS\""
  rsync -ah --progress "$POD:/workspace/backup_models/eval_results_t141_shared_x_pes/" "$BM/eval_results_t141_shared_x_pes/"
  rsync -ah --progress "$POD:/workspace/backup_models/logs/t141_*" "$BM/logs/"
  exit 0
fi

# Local execution. GPU pinning when --parallel-gpus is set.
# GPU-busy guard: bracket-trick so the pgrep pattern doesn't self-match this shell.
# Skip when CUDA_VISIBLE_DEVICES is already set by an outer orchestrator (e.g. when
# running ablation on GPU 0 in parallel with EAC on GPU 1 — pgrep would false-positive
# on the sister job's llama-server).
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if pgrep -f "[l]m-eval|[l]lama-server|[o]mk_eval" >/dev/null 2>&1; then
    echo "FATAL: another eval/server is already running — refusing to start."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
    exit 1
  fi
fi
export PATH=/root/anaconda3/envs/omnimergekit/bin:/opt/llama.cpp/build/bin:$PATH
export HF_ALLOW_CODE_EVAL=1 OMK_NO_README=1
TS=$(date +%Y%m%d_%H%M%S)
echo "[t141] START $(date -Iseconds)  cells={$CELLS}" | tee "$BM/logs/t141_dispatch.log"

if [ -n "$PARALLEL_GPUS" ]; then
  IFS=',' read -ra GPUS <<< "$PARALLEL_GPUS"
  declare -i gi=0
  declare -a PIDS=()
  for c in $CELLS; do
    g=${GPUS[$((gi % ${#GPUS[@]}))]}
    echo "  [parallel] cell $c -> GPU $g" | tee -a "$BM/logs/t141_dispatch.log"
    (CUDA_VISIBLE_DEVICES=$g build_cell "$c") &
    PIDS+=($!)
    gi=$((gi + 1))
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || true; done
else
  for c in $CELLS; do
    build_cell "$c" || echo "  [t141] cell $c had errors (continuing)"
  done
fi

echo "[t141] DONE $(date -Iseconds)" | tee -a "$BM/logs/t141_dispatch.log"
touch "$BM/logs/T141_ABLATION_${TS}_DONE"

# ============================ 2×2 summary table ============================
RES=$BM/eval_results_t141_shared_x_pes
echo
echo "=================== T141 2×2 ABLATION RESULTS ==================="
echo "  rows = shared α, cols = per-expert α; cell = HE+164 / MPE-300 score"
echo
for tpl in $EVALS; do
  echo "  --- $tpl ---"
  echo "  | shared\\pes | p=1.0 | p=1.20 |"
  echo "  |-----------|-------|--------|"
  for s in A B; do
    sval=$( [ $s = A ] && echo "1.0" || echo "1.2" )
    row="  | s=$sval     "
    for p in 1 2; do
      cell="${s}${p}"
      sj=$RES/$tpl/t141-62e-${cell}/summary.json
      if [ -f "$sj" ]; then
        score=$("$PY" -c "import json;print(round(json.load(open('$sj')).get('score',0)*100,2))")
        row+=$(printf " | %5s%%" "$score")
      else
        row+=" | -----"
      fi
    done
    echo "${row} |"
  done
  echo
done | tee -a "$BM/logs/t141_dispatch.log"
