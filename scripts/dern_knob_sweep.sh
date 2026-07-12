#!/usr/bin/env bash
# dern_knob_sweep.sh — T206 DERN knob sweep on BARE dern11 (re-pointed after the eogfk combo
# regressed: combo Q6_K-noimat 9/48@t0.9 vs dern11 4/48). Base = published v7-coder 98e student +
# C6v3lcb keep-meta (set via env for the worker). Waits for both GPUs to free, then runs 4 variants
# in 2 rounds (2-GPU parallel). All Q6_K noimat, gated vendor_minp_rep {0.9, 0.8}; one variable each
# vs the existing dern11 baseline (solar / survivor / freq^1.0 = 4/48@t0.9, 2/48@t0.8):
#   cs-bal  : corpus = 9bench-balanced     (corpus axis)
#   an-mean : norm-anchor = members_mean    (anchor axis)
#   fx-2.0  : freq-exponent = 2.0 (sharpen) (freq axis)
#   fx-0.5  : freq-exponent = 0.5 (flatten) (freq axis)
set -uo pipefail
SFT=/mnt/sdc/ml/sft_heal
W=/srv/ml/scripts/dern_knob_variant.sh
SOLAR=$SFT/eog_corpus_solar.jsonl
BAL=$SFT/eog_corpus_solar_9bench_mix.jsonl
# BARE dern11 base
export STUDENT=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-it
export KEEPMETA=$SFT/v7coder_C6v3lcb_keepmeta.json
export TAG=dern11-knob
ts(){ date '+%T %Z'; }
echo "==================== dern_knob_sweep (base=BARE dern11) $(ts) ===================="
for f in "$W" "$SOLAR" "$BAL" "$STUDENT/config.json" "$KEEPMETA"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

# wait until both GPUs are free (current combo + imat gates finished)
echo "[$(ts)] waiting for both GPUs to free ..."
while true; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
  [ "${busy:-0}" -eq 0 ] && break
  sleep 120
done
echo "[$(ts)] both GPUs free — starting sweep on BARE dern11"

run_round() {  # <labelA corpusA anchorA fexpA> <labelB corpusB anchorB fexpB>
  bash "$W" "$1" "$2" "$3" "$4" 0 8190 > "$SFT/dern_sweep_$1.log" 2>&1 &
  local pA=$!
  bash "$W" "$5" "$6" "$7" "$8" 1 8191 > "$SFT/dern_sweep_$5.log" 2>&1 &
  local pB=$!
  wait "$pA"; local rA=$?
  wait "$pB"; local rB=$?
  echo "[$(ts)] round done: $1 rc=$rA | $5 rc=$rB"
}

# Round 1: corpus axis (GPU0) + anchor axis (GPU1)
run_round  cs-bal  "$BAL"   survivor      1.0   an-mean "$SOLAR" members_mean 1.0
# Round 2: freq sharpen (GPU0) + freq flatten (GPU1)
run_round  fx-2.0  "$SOLAR" survivor      2.0   fx-0.5  "$SOLAR" survivor     0.5

echo "[$(ts)] === DERN KNOB SWEEP DONE ==="
echo "===== ROLL-UP (loops/48 per arm) — baseline dern11 noimat = 4/48@t0.9, 2/48@t0.8 ====="
for v in cs-bal an-mean fx-2.0 fx-0.5; do
  echo "--- $v ---"
  grep -E "fails=|arm summary" "$SFT/dern_sweep_$v.log" 2>/dev/null | tail -4 || echo "  (no summary)"
done
