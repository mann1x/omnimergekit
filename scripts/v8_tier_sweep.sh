#!/usr/bin/env bash
# v8_tier_sweep.sh — BUILD-then-EVAL per-tier HE+/MPE sweep for the v8
# (fkbroad-soft2) GGUF ladder that overwrites v7-coder. Unlike quant_sweep_v7.sh
# (which downloads already-published tiers), these tiers don't exist yet, so each
# cell:  quantize_gguf --only <tier> (CPU; reuses pre-staged F16 + imatrix, no GPU
# recompute) -> omk_eval HE+ & MPE (GPU) -> delete the tier gguf. Disk-bounded:
# never more than F16 + 1 tier per instance. Lock/done idempotent + 2-instance safe
# (launch a 2nd copy on the other GPU; they split remaining tiers). Resumable.
#
# Usage:  v8_tier_sweep.sh <GPU> <PORT>
#   e.g.  v8_tier_sweep.sh 0 8250      # GPU0 now
#         v8_tier_sweep.sh 1 8251      # GPU1 after the card LCB evals free it
set -uo pipefail
GPU="${1:?usage: v8_tier_sweep.sh <GPU> <PORT>}"
PORT="${2:?usage: v8_tier_sweep.sh <GPU> <PORT>}"

PY=/root/anaconda3/envs/omnimergekit/bin/python
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
QUANTIZE=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TPL_DIR=/srv/ml/repos/omnimergekit/eval/templates
BASE_TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
LLAMA_BIN_EVAL=/mnt/sdc/ml/llama.cpp-b9700/build/bin        # eval binary (matches v8 suite)

SFT=/mnt/sdc/ml/sft_heal
BF16=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-it
MODEL_NAME=gemma-4-A4B-98e-v7-coder-fkbroad-soft2-it
F16SRC=$SFT/fkbroad-soft2-F16.gguf
IMAT=$SFT/fkbroad-soft2-imatrix.dat
BMID=ManniX-ITA/gemma-4-A4B-98e-v7-coder-it                 # base_model id for README (not uploaded; --no-upload)

ROOT=/mnt/sdc/ml/v8_tier_sweep
OUTDIR=$ROOT/gguf
RESULTS=$ROOT/results
WORK=$ROOT/work
mkdir -p "$OUTDIR" "$RESULTS" "$WORK"
LOG="$WORK/sweep_g${GPU}_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

BENCHES=(humanevalplus_full multipl_e_100)
# exact published-ladder set minus Q6_K (already validated: HE+ 93.29 / MPE 89.33);
# qat-Q4_0 + CD-qat-Q4_K_M are a SEPARATE stage (need the v8-on-QAT-base sub-build).
TIERS=(
  Q8_0 Q6_K_L Q5_K_L Q5_K_M Q5_K_S Q4_K_L Q4_K_M Q4_K_S Q4_0 Q4_1
  IQ4_NL IQ4_XS Q3_K_XL Q3_K_L Q3_K_M Q3_K_S IQ3_M Q2_K_L IQ2_XS
  CD-Q6_K CD-Q5_K_M CD-Q4_K_M CD-Q3_K_L CD-Q2_K
)

ts(){ date -u '+%T'; }
echo "[sweep $(ts)] START gpu=$GPU port=$PORT  tiers=${#TIERS[@]}  out=$OUTDIR"

# ---- preflight ----
for f in "$QUANTIZE" "$OMK" "$F16SRC" "$IMAT" "$BASE_TOK/tokenizer.json" \
         "$LLAMA_BIN_EVAL/llama-server" "$BF16/model.safetensors"; do
  [ -e "$f" ] || { echo "[FATAL] missing $f"; exit 1; }
done
for b in "${BENCHES[@]}"; do
  [ -f "$TPL_DIR/$b.yaml" ] || { echo "[FATAL] template missing: $b"; exit 1; }
done
# stage F16 + imatrix into OUTDIR under the names quantize_gguf expects (reuse, no recompute)
[ -e "$OUTDIR/${MODEL_NAME}-F16.gguf" ] || ln -s "$F16SRC" "$OUTDIR/${MODEL_NAME}-F16.gguf"
[ -e "$OUTDIR/imatrix.dat" ] || ln -s "$IMAT" "$OUTDIR/imatrix.dat"
echo "[sweep $(ts)] preflight OK; staged F16+imatrix in OUTDIR"

score_of(){ "$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('score'))" "$1" 2>/dev/null; }

do_cell(){
  local tier="$1"
  local served="v8-${tier}"
  local done="$WORK/${tier}.done" lock="$WORK/${tier}.lock"
  [ -e "$done" ] && return 0
  # resume: both summaries already present
  local have=1
  for b in "${BENCHES[@]}"; do [ -f "$RESULTS/$b/$served/summary.json" ] || have=0; done
  if [ "$have" -eq 1 ]; then echo "[done-resume $(ts)] $served"; : >"$done"; return 0; fi
  mkdir "$lock" 2>/dev/null || return 0          # another instance owns this tier
  trap 'rmdir "$lock" 2>/dev/null || true' RETURN

  local gguf="$OUTDIR/${MODEL_NAME}-${tier}.gguf"
  echo "==== [$served] $(ts) gpu=$GPU ===="

  # --- BUILD (CPU; reuses pre-staged F16 + imatrix) ---
  if [ ! -f "$gguf" ]; then
    echo "[build $(ts)] quantize_gguf --only $tier (no-upload)"
    "$PY" "$QUANTIZE" --model "$BF16" --output-dir "$OUTDIR" \
        --only "$tier" --no-upload --keep-local --base-model-id "$BMID" \
        > "$WORK/${tier}.build.log" 2>&1
    local brc=$?
    if [ $brc -ne 0 ] || [ ! -f "$gguf" ]; then
      echo "[ERR $(ts)] build failed $tier rc=$brc"; tail -8 "$WORK/${tier}.build.log"; return 1
    fi
    # sanity: pre-staged imatrix reused (no silent recompute), GGUF magic OK
    grep -q "pre-staged imatrix" "$WORK/${tier}.build.log" 2>/dev/null \
      && echo "[build $(ts)] (imatrix reused)" || echo "[build $(ts)] WARN: pre-stage line not seen"
    [ "$("$PY" -c "print(open('$gguf','rb').read(4).decode('latin1'))" 2>/dev/null)" = "GGUF" ] \
      || { echo "[ERR] bad GGUF header $tier"; return 1; }
    echo "[build $(ts)] $tier ready $(du -h "$gguf" | cut -f1)"
  fi

  # --- EVAL HE+ then MPE ---
  local ok=1
  for b in "${BENCHES[@]}"; do
    [ -f "$RESULTS/$b/$served/summary.json" ] && { echo "[skip] $b/$served $(score_of "$RESULTS/$b/$served/summary.json")"; continue; }
    echo "[eval $(ts)] $b  $served"
    LLAMA_BIN="$LLAMA_BIN_EVAL" CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$OMK" \
        --model "$gguf" --tokenizer "$BASE_TOK" \
        --template "$b" --backend llama --quant gguf \
        --served-name "$served" --results-dir "$RESULTS" --port "$PORT"
    if [ $? -ne 0 ] || [ ! -f "$RESULTS/$b/$served/summary.json" ]; then
      echo "[ERR $(ts)] $b/$served (no summary)"; ok=0; break
    fi
    echo "[ok $(ts)] $b/$served score=$(score_of "$RESULTS/$b/$served/summary.json")"
  done

  if [ "$ok" -eq 1 ]; then
    : >"$done"
    rm -f "$gguf"                                 # bound disk: keep F16 + results, drop tier
    echo "[CELL-DONE $(ts)] $served  HE+=$(score_of "$RESULTS/humanevalplus_full/$served/summary.json")  MPE=$(score_of "$RESULTS/multipl_e_100/$served/summary.json")  (gguf deleted)"
  else
    echo "[CELL-FAIL $(ts)] $served — leaving gguf for retry"
  fi
}

for tier in "${TIERS[@]}"; do do_cell "$tier"; done
echo "###### V8_TIER_SWEEP_DONE g${GPU} $(ts) ######"
# roll-up
echo "===== v8 tier table (this instance's view) ====="
printf "%-12s %8s %8s\n" tier HE+ MPE
for tier in Q6_K "${TIERS[@]}"; do
  he=$(score_of "$RESULTS/humanevalplus_full/v8-${tier}/summary.json")
  mp=$(score_of "$RESULTS/multipl_e_100/v8-${tier}/summary.json")
  printf "%-12s %8s %8s\n" "$tier" "${he:-–}" "${mp:-–}"
done
