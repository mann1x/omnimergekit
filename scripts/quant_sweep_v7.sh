#!/usr/bin/env bash
# quant_sweep_v7.sh — per-tier HE+ & MPE100 sweep across the published v7-coder /
# v7-coderx GGUF ladders (29 tiers each), on bs2. Greedy/canonical, llama backend.
#
# Design goals:
#   - Disk-bounded: download ONE tier gguf, eval both benches, DELETE it. Never
#     holds more than (n_instances) tiers at once.
#   - Shardable & resumable WITHOUT coordination: each (model,tier) cell is claimed
#     atomically via `mkdir <cell>.lock`; completion via `<cell>.done`. Launch a
#     second copy on another GPU and the two instances naturally split remaining
#     work. Re-launching after a crash resumes (existing summary.json => done).
#   - GPU-pinned via CUDA_VISIBLE_DEVICES so it coexists with the GPU0 F16 suite.
#
# Usage:  quant_sweep_v7.sh <GPU> <PORT>
#   e.g.  quant_sweep_v7.sh 1 8240        # GPU1 instance (start now)
#         quant_sweep_v7.sh 0 8241        # GPU0 instance (after F16 + LCB-retry)
set -uo pipefail

GPU="${1:?usage: quant_sweep_v7.sh <GPU> <PORT>}"
PORT="${2:?usage: quant_sweep_v7.sh <GPU> <PORT>}"

PY=/srv/ml/envs/envs/omnimergekit/bin/python
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH   # hf CLI + tooling from omk env
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TPL_DIR=/srv/ml/repos/omnimergekit/eval/templates
RESULTS=/srv/ml/eval_results_v7_quant_sweep
GGUF_DIR=/mnt/sdc/ml/quant_sweep_gguf
WORK=/srv/ml/scripts/quant_sweep_work
mkdir -p "$RESULTS" "$GGUF_DIR" "$WORK"
LOG="$WORK/quant_sweep_g${GPU}_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

BENCHES=(humanevalplus_full multipl_e_100)

# model spec:  served_base | hf_repo | filename_prefix | tokenizer_dir
MODELS=(
  "v7coder|ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF|gemma-4-A4B-98e-v7-coder-it-|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it"
  "v7coderx|ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it-GGUF|gemma-4-A4B-98e-v7-coderx-it-|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it"
)

# ladder first (most informative), then the rest. 8 + 21 = 29.
LADDER=(Q8_0 Q6_K Q5_K_M Q4_K_M IQ4_XS IQ3_XS IQ2_S CD-Q4_K_M)
REST=(CD-IQ4_K_M CD-Q3_K_L CD-Q5_K_M CD-Q6_K IQ2_M IQ2_XS IQ3_M IQ3_XXS IQ4_NL \
      Q2_K_L Q3_K_L Q3_K_M Q3_K_S Q3_K_XL Q4_0 Q4_1 Q4_K_L Q4_K_S Q5_K_L Q5_K_S Q6_K_L)
TIERS=("${LADDER[@]}" "${REST[@]}")

echo "[sweep] START $(date -u)  gpu=$GPU port=$PORT  tiers=${#TIERS[@]}  models=${#MODELS[@]}"

# ---- preflight ----
[ -f "$OMK" ] || { echo "[FATAL] omk_eval not found: $OMK"; exit 1; }
for b in "${BENCHES[@]}"; do
  [ -f "$TPL_DIR/$b.yaml" ] || { echo "[FATAL] template missing: $TPL_DIR/$b.yaml"; exit 1; }
done
for m in "${MODELS[@]}"; do
  IFS='|' read -r base repo prefix tok <<<"$m"
  [ -d "$tok" ] || { echo "[FATAL] tokenizer dir missing for $base: $tok"; exit 1; }
done
command -v hf >/dev/null 2>&1 || { echo "[FATAL] hf CLI not on PATH"; exit 1; }
echo "[sweep] preflight OK"

score_of() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['score'])" "$1" 2>/dev/null; }

do_cell() {
  local base="$1" repo="$2" prefix="$3" tok="$4" tier="$5"
  local served="${base}-${tier}"
  local cell="${base}__${tier}"
  local done="$WORK/$cell.done" lock="$WORK/$cell.lock"

  [ -e "$done" ] && return 0

  # idempotent resume: if both benches already have a score, mark done.
  local have=1
  for b in "${BENCHES[@]}"; do
    [ -f "$RESULTS/$b/$served/summary.json" ] || have=0
  done
  if [ "$have" -eq 1 ]; then echo "[done-resume] $served (both summaries present)"; : >"$done"; return 0; fi

  mkdir "$lock" 2>/dev/null || { return 0; }   # another instance owns this cell
  trap 'rmdir "$lock" 2>/dev/null || true' RETURN

  local fname="${prefix}${tier}.gguf"
  local gguf="$GGUF_DIR/$fname"
  echo "==== [$served] $(date -u) gpu=$GPU ===="

  if [ ! -f "$gguf" ]; then
    echo "[dl  ] $repo :: $fname"
    if ! hf download "$repo" "$fname" --local-dir "$GGUF_DIR" >/dev/null 2>"$WORK/$cell.dl.err"; then
      echo "[ERR ] download failed for $served"; sed -n '1,5p' "$WORK/$cell.dl.err"; return 1
    fi
  fi
  [ -f "$gguf" ] || { echo "[ERR ] gguf absent after download: $gguf"; return 1; }

  local ok=1
  for b in "${BENCHES[@]}"; do
    if [ -f "$RESULTS/$b/$served/summary.json" ]; then
      echo "[skip] $b/$served already scored ($(score_of "$RESULTS/$b/$served/summary.json"))"; continue
    fi
    echo "[eval] $b  served=$served"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$OMK" \
        --model "$gguf" --tokenizer "$tok" \
        --template "$TPL_DIR/$b.yaml" --backend llama \
        --served-name "$served" --results-dir "$RESULTS" --port "$PORT"
    local rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$RESULTS/$b/$served/summary.json" ]; then
      echo "[ERR ] $b/$served rc=$rc (no summary)"; ok=0; break
    fi
    echo "[ok  ] $b/$served score=$(score_of "$RESULTS/$b/$served/summary.json")"
  done

  if [ "$ok" -eq 1 ]; then
    : >"$done"
    rm -f "$gguf"                       # bound disk: drop the tier, keep results
    echo "[CELL-DONE] $served  HE+=$(score_of "$RESULTS/humanevalplus_full/$served/summary.json")  MPE=$(score_of "$RESULTS/multipl_e_100/$served/summary.json")  (gguf deleted)"
  else
    echo "[CELL-FAIL] $served — leaving gguf for retry, no done marker"
  fi
}

for tier in "${TIERS[@]}"; do
  for m in "${MODELS[@]}"; do
    IFS='|' read -r base repo prefix tok <<<"$m"
    do_cell "$base" "$repo" "$prefix" "$tok" "$tier"
  done
done

echo "###### QUANT_SWEEP_DONE g${GPU} $(date -u) ######"
