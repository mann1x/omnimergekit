#!/usr/bin/env bash
# gate_v8_q4km.sh — T224: loop-gate the v8 DEPLOYMENT tier (imat-Q4_K_M).
#
# The shipping v8 (gemma-4-A4B-98e-v7-coder-fkbroad-soft2) loop verdict — "keep the
# average DERN fold, 0/48 loops" — was established ONLY at imat-Q6. Q4_K_M is the tier
# users actually pull, and the HF complaint that started the whole loop saga (#575) was
# literally "v7-coder Q4_K_M llama.cpp looping". The no-imat-Q4_0 probe (9/48·12/48)
# proved low-bit tiers CAN reintroduce loops the imat-Q6 never shows. This gates the
# real deployment tier: imat-Q4_K_M, same recipe the published K-quants ship with.
#
# Cheap path: the Jun-19 build LEFT the F16 + the preserved imatrix on disk (the build
# script's rm -f $F16 never fired), so this is a ~3-5 min CPU quant off F16 — NOT a DERN
# rebuild. imatrix is the SOURCE (already preserved), so the archival rule is satisfied.
# GPU0, llama.cpp-latest (-c 131072 → gate cannot ctx-overflow). PID-kill only.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
LCPP=/mnt/sdc/ml/llama.cpp-latest
SFT=/mnt/sdc/ml/sft_heal
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
AL=/srv/ml/agentic_loop

F16=$SFT/fkbroad-soft2-F16.gguf
IMAT=$SFT/fkbroad-soft2-imatrix.dat                       # PRESERVED (source, not rebuilt)
Q4KM=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-imat-Q4_K_M.gguf
GRES=$AL/results/fkbroad-soft2-imatQ4KM_minp48.json
GPU=0
PORT=8264
LABEL=fkbroad-soft2-imatQ4KM
ts(){ date '+%T %Z'; }
echo "==================== T224 v8 imat-Q4_K_M loop gate $(ts) ===================="

# ── preflight ─────────────────────────────────────────────
for f in "$F16" "$IMAT" "$LCPP/build/bin/llama-quantize" "$GATE" \
         "$AL/fixtures/solar_build_start.json"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
magic=$("$PY" -c "print(open('$F16','rb').read(4).decode('latin1'))" 2>/dev/null)
[ "$magic" = "GGUF" ] || { echo "FATAL F16 bad header ($magic)"; exit 2; }
free=$(df --output=avail -BG "$SFT" | tail -1 | tr -dc '0-9')
echo "[preflight $(ts)] ${free}G free on $SFT; imatrix $(stat -c%s "$IMAT") bytes (PRESERVED source)"
[ "${free:-0}" -lt 15 ] && { echo "FATAL <15G free — Q4_K_M write needs ~11G"; exit 9; }

# ── 1. quant imat-Q4_K_M (CPU, ~3-5 min) ─────────────────
if [ ! -f "$Q4KM" ]; then
  echo "[1 $(ts)] llama-quantize --imatrix Q4_K_M -> $Q4KM"
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q4KM" Q4_K_M 32 \
    || { echo "FATAL quant Q4_K_M"; exit 10; }
else echo "[1] $Q4KM exists, skip quant"; fi
qmagic=$("$PY" -c "print(open('$Q4KM','rb').read(4).decode('latin1'))" 2>/dev/null)
[ "$qmagic" = "GGUF" ] || { echo "FATAL Q4_K_M bad header ($qmagic)"; exit 10; }
echo "[1 $(ts)] Q4_K_M $(du -h "$Q4KM"|cut -f1) OK"

# ── 2. 48-seed loop gate vendor_minp_rep {0.9,0.8} ───────
echo "[2 $(ts)] loop gate $LABEL GPU$GPU:$PORT  vs imat-Q6 0/48,0/48"
bash "$GATE" "$Q4KM" "$GPU" "$PORT" "$GRES" "$LABEL" || echo "[2] WARN gate rc=$?"

echo "[3 $(ts)] === T224 v8 imat-Q4_K_M GATE DONE ==="
[ -f "$GRES" ] && "$PY" -c "import json;d=json.load(open('$GRES'));[print('  ',r['config'],'fails',r['fails'],'/',r['seeds'],'loops',r.get('loops',round(r.get('loop_rate',0)*r['seeds']))) for r in d['results']]" 2>/dev/null
echo "imat-Q4_K_M: $Q4KM ; imatrix(preserved): $IMAT"
echo "###### T224_Q4KM_GATE_DONE $(ts) ######"
