#!/usr/bin/env bash
# gate_pub_v7q6.sh [GPU] [PORT] — download the PUBLISHED v7-coder Q6_K GGUF and
# run it through the 48-seed agentic loop gate (vendor_minp_rep x t{0.9,0.8},
# --reasoning-budget 12288). This is the published->loopfix before-number.
set -uo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
HF=/root/anaconda3/envs/omnimergekit/bin/hf
SFT=/mnt/sdc/ml/sft_heal
DLDIR=$SFT/pub_v7_gguf
FILE=gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf
PUB=$DLDIR/$FILE
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
OUT=/srv/ml/agentic_loop/results/pub_v7_q6_minp48.json
GPU=${1:-1}; PORT=${2:-8191}
ts(){ date '+%T %Z'; }
echo "[pubq6 $(ts)] download (if absent) -> gate GPU$GPU:$PORT"
for f in "$HF" "$GATE"; do [ -e "$f" ] || { echo "[pubq6] FATAL missing $f"; exit 9; }; done
[ -f "$PUB" ] || "$HF" download ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF "$FILE" \
  --repo-type model --local-dir "$DLDIR" || { echo "[pubq6] FATAL download"; exit 2; }
[ -f "$PUB" ] || { echo "[pubq6] FATAL pub gguf missing after download"; exit 3; }
ls -la "$PUB"
bash "$GATE" "$PUB" "$GPU" "$PORT" "$OUT" pub-v7-q6
echo "[pubq6 $(ts)] PUBQ6_GATE_DONE  out=$OUT"
