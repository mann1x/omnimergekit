#!/usr/bin/env bash
# build_v9_q40_gate.sh — v9-agentic no-imatrix Q4_0 + the canonical 48-seed loop gate.
#
# WHY no-imat Q4_0 and NOT Q6_K: the T223 campaign gated every cell at this tier because
# it is the DISCRIMINATING one. Measured there, CTRL (no force-keep) loops 31-36/48 while
# STD16 (46 agentic_eog pins) loops 0-1/48. With an imatrix at Q6 the whole family reads
# 0/48 and the gate separates nothing. Same binary (b9700), same matrix, same fixture, so
# v9 drops straight into the existing table.
#
# v9's force-keep is the <turn|> (id 106, DELIVERY close) top-4/layer lift set, 120 pins,
# vs STD16's 46 pins on <|tool_response> (id 50, PER-STEP stop). Overlap between the two
# pin sets is exactly 0, so this is a genuine head-to-head of which terminator to protect.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
# NOTE: T223 quantized with llama.cpp-latest, which now symlinks to b10088 and has no
# llama-quantize. Using b9700 -- the SAME build as the gate server -- and recording the
# deviation rather than silently picking a different one.
LCPP=/mnt/sdc/ml/llama.cpp-b9700
GG=/mnt/sdc/ml/v9build/v9-agentic-GGUF
F16=$GG/v9-agentic-bf16-F16.gguf
Q40=$GG/v9-agentic-Q4_0-noimat.gguf
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh
AL=/srv/ml/agentic_loop
OUT=$AL/results/t223_V9AGENTIC_Q40_minp48.json
ts(){ date -u '+%H:%M:%SZ'; }

echo "===== V9 Q4_0 + LOOP GATE $(ts) ====="
for f in "$F16" "$GATE" "$LCPP/build/bin/llama-quantize" "$AL/fixtures/solar_build_start.json" \
         /mnt/sdc/ml/llama.cpp-b9700/build/bin/llama-server; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

if [ ! -f "$Q40" ]; then
  echo "[q40 $(ts)] quantizing Q4_0 with NO imatrix (discriminating tier)"
  "$LCPP/build/bin/llama-quantize" "$F16" "$Q40" Q4_0 32 || { echo "FATAL quant"; exit 1; }
  magic=$("$PY" -c "print(open('$Q40','rb').read(4).decode('latin1'))" 2>/dev/null)
  [ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF"; exit 1; }
fi
echo "[q40 $(ts)] Q4_0 ready: $(stat -c%s "$Q40") bytes"
# assert NO imatrix KV -- this tier must be uncalibrated or it is not comparable to T223
"$PY" - "$Q40" <<'PY'
import sys
from gguf import GGUFReader
have = {f.name for f in GGUFReader(sys.argv[1]).fields.values()}
imat = [k for k in have if k.startswith("quantize.imatrix")]
print("  imatrix KV present: %s" % (imat or "NONE (correct for this gate)"))
sys.exit(1 if imat else 0)
PY
[ $? -eq 0 ] || { echo "FATAL Q4_0 carries an imatrix — not comparable to the T223 cells"; exit 1; }

echo "[gate $(ts)] 48 seeds x {minp t0.9, t0.8} on GPU0 -> $OUT"
MATRIX=matrix_minp_2temp.json bash "$GATE" "$Q40" 0 8410 "$OUT" "v9agentic-q40" \
  > "$AL/logs/t223_V9AGENTIC_gate.log" 2>&1
echo "[gate $(ts)] GATE_RC=$?"

echo "[table $(ts)] === T223 loop table WITH v9 ==="
"$PY" - <<'PY'
import json, os
AL="/srv/ml/agentic_loop/results"
order=[("CTRL","fkbroad, NO force-keep"),("STD8","8 pins id50"),
       ("STD16","46 pins id50 = PUBLISHED"),("STD27","27 pins id50"),
       ("ADD16","STD16+DERN"),("V8","DERN, no force-keep"),
       ("V9AGENTIC","120 pins id106 + agentic map")]
print("%-11s %-26s %16s %16s" % ("cell","recipe","t0.9 fail/loop","t0.8 fail/loop"))
for nm,desc in order:
    p=f"{AL}/t223_{nm}_Q40_minp48.json"
    if not os.path.exists(p):
        print("%-11s %-26s %16s" % (nm,desc,"(no result)")); continue
    d=json.load(open(p)); row={}
    for r in d.get("results",[]):
        cfg=str(r.get("config","")); t="t0.9" if "0.9" in cfg else "t0.8"
        row[t]="f%s/l%s" % (r.get("fails","?"), r.get("loops","?"))
    print("%-11s %-26s %16s %16s" % (nm,desc,row.get("t0.9","-"),row.get("t0.8","-")))
PY
echo "###### V9_GATE_DONE $(ts) ######"
