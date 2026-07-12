#!/usr/bin/env bash
# gate_tiers_v8.sh <GPU> <PORT> <RESULTS_TSV> <NATIVE_TIER...>
# Loop-gate the published v8 (fkbroad-soft2) NATIVE-type quant tiers: build each from the
# preserved F16 + imatrix via raw llama-quantize, run the canonical 48-seed agentic gate
# (gate_sweep48_minp_p.sh, matrix_minp_2temp.json = vendor_minp_rep x {temp 0.9, 0.8}),
# record loops/48 per temp, then DELETE the tier gguf (disk hygiene; F16+imatrix are the
# preserved sources and stay). Same gate script that produced Q6_K 0/48 + Q4_K_M 2/48, so
# the new rows are directly comparable. PID-kill only (gate_sweep48 traps its own server).
#
# NATIVE tiers only (raw llama-quantize ftype names). The _L mixes / CD-* / qat tiers need
# quantize_gguf.py and are a separate phase-2 pass.
set -uo pipefail
GPU="$1"; PORT="$2"; TSV="$3"; shift 3
SFT=/mnt/sdc/ml/sft_heal
F16=$SFT/fkbroad-soft2-F16.gguf
IMAT=$SFT/fkbroad-soft2-imatrix.dat
LQ=/mnt/sdc/ml/llama.cpp-latest/build/bin/llama-quantize
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
WORK=/mnt/sdc/ml/v8_loopsweep
PY=/root/anaconda3/envs/omnimergekit/bin/python
mkdir -p "$WORK/results" "$(dirname "$TSV")"
ts(){ date '+%T %Z'; }

# preflight
for f in "$F16" "$IMAT" "$LQ" "$GATE" /srv/ml/agentic_loop/fixtures/solar_build_start.json; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
echo "[$(ts)] GPU$GPU:$PORT sweep tiers: $*"

for TIER in "$@"; do
  OUT=$WORK/results/loop_${TIER}.json
  GGUF=$WORK/soft2-${TIER}.gguf
  # skip if already recorded
  if grep -q "^${TIER}	" "$TSV" 2>/dev/null; then echo "[$(ts)] $TIER already in TSV, skip"; continue; fi
  # disk guard: need ~22G headroom for the biggest tier
  free=$(df --output=avail -BG "$SFT" | tail -1 | tr -dc '0-9')
  [ "${free:-0}" -lt 25 ] && { echo "[$(ts)] $TIER SKIP: only ${free}G free"; echo "$TIER	LOWDISK_${free}G" >> "$TSV"; continue; }

  echo "[$(ts)] === $TIER : build (GPU$GPU) ==="
  if [ ! -f "$GGUF" ]; then
    "$LQ" --imatrix "$IMAT" "$F16" "$GGUF" "$TIER" 32 > "$WORK/results/build_${TIER}.log" 2>&1 \
      || { echo "[$(ts)] $TIER BUILD_FAIL"; echo "$TIER	BUILD_FAIL" >> "$TSV"; rm -f "$GGUF"; continue; }
  fi
  magic=$("$PY" -c "import sys;print(open(sys.argv[1],'rb').read(4).decode('latin1'))" "$GGUF" 2>/dev/null)
  [ "$magic" = "GGUF" ] || { echo "[$(ts)] $TIER BAD_HEADER($magic)"; echo "$TIER	BAD_HEADER" >> "$TSV"; rm -f "$GGUF"; continue; }
  sz=$(du -h "$GGUF" | cut -f1)

  echo "[$(ts)] === $TIER : 48-seed gate (GPU$GPU:$PORT, $sz) ==="
  bash "$GATE" "$GGUF" "$GPU" "$PORT" "$OUT" "soft2-$TIER" > "$WORK/results/gate_${TIER}.log" 2>&1 \
    || echo "[$(ts)] $TIER gate rc=$?"

  if [ -f "$OUT" ]; then
    line=$("$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(' '.join('%s=%d/%d'%(r['config'],r['fails'],r['seeds']) for r in d['results']))" "$OUT" 2>/dev/null)
    echo "$TIER	${sz}	${line:-PARSE_FAIL}" >> "$TSV"
    echo "[$(ts)] $TIER -> ${line}"
  else
    echo "$TIER	${sz}	NO_OUT" >> "$TSV"
    echo "[$(ts)] $TIER NO_OUT (gate produced no json)"
  fi
  rm -f "$GGUF"
done
echo "[$(ts)] ===== GPU$GPU TIER SWEEP DONE ====="
