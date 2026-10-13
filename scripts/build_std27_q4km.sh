#!/usr/bin/env bash
# build_std27_q4km.sh — STD27-v5 Q4_K_M for deployment, quantized with the calib_v5 imatrix.
#
# Source is the F16, never the existing Q6_K — re-quantizing a quant is double quantization.
# The imatrix is STD27-v5's OWN calib_v5 one (NOT the calib_both leftover in t223_fk, and NOT
# the v9-agentic AC imatrix): this tier must match the Q6_K that the LCB arm is measuring.
#
# THREADS is deliberately 16, not 32: the STD27 LCB eval is running concurrently and its
# llama-server needs cores. Quantize is CPU-only, so it cannot touch the GPU arm.
set -uo pipefail
ts() { date -u +%H:%M:%SZ; }

LCPP=/mnt/sdc/ml/llama.cpp-b9700
GG=/mnt/sdc/ml/std27v5/STD27-v5-GGUF
F16="$GG/STD27-v5-bf16-F16.gguf"
IMAT="$GG/imatrix.dat"
NAME=gemma-4-A4B-98e-v9-agentic-std27-it-Q4_K_M.gguf
OUT="$GG/$NAME"
THREADS="${THREADS:-16}"

for f in "$LCPP/build/bin/llama-quantize" "$F16" "$IMAT"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
[ -f "$OUT" ] && { echo "FATAL: $OUT exists — refusing to clobber"; exit 3; }

MAGIC=$(head -c 4 "$IMAT")
[ "$MAGIC" = "GGUF" ] || { echo "FATAL: imatrix not GGUF (magic=$MAGIC)"; exit 4; }

# Assert this is the calib_v5 imatrix, not calib_both and not the AC one.
if ! /root/anaconda3/envs/omnimergekit/bin/python - "$IMAT" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
f = r.fields["imatrix.datasets"]
ds = bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
print("[imatrix] dataset=%s entries=%d" % (ds, len(r.tensors)))
if "calibration_datav5.txt" not in ds:
    sys.exit("FATAL: expected calibration_datav5.txt, got %s" % ds)
PY
then
  exit 5
fi

df -h /mnt/sdc | tail -1
echo "[$(ts)] quantize Q4_K_M (calib_v5 imatrix, $THREADS threads) -> $NAME"
"$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$OUT" Q4_K_M "$THREADS" 2>&1 | tail -4
[ -s "$OUT" ] || { echo "[$(ts)] FATAL: produced nothing"; exit 6; }

echo "[$(ts)] $NAME $(stat -c%s "$OUT") bytes"
sha256sum "$OUT" | awk -v n="$NAME" '{print $1"  "n}' | tee "$OUT.sha256"

# Prove the imatrix rode in — a GGUF KV fact, not a flag we passed.
if ! /root/anaconda3/envs/omnimergekit/bin/python - "$OUT" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
hits = {}
for k, f in r.fields.items():
    if "imatrix" in k.lower():
        try:
            hits[k] = bytes(f.parts[f.data[0]]).decode("utf-8", "replace") \
                      if f.types and f.types[0] == 8 else f.parts[f.data[0]].tolist()
        except Exception:
            hits[k] = "<?>"
if not hits:
    sys.exit("FATAL: no imatrix KV — quantized WITHOUT it")
for k in sorted(hits):
    print("   %-38s %s" % (k, str(hits[k])[:110]))
PY
then
  exit 7
fi

# The served tier must carry the same template + terminator as the measured Q6_K.
/root/anaconda3/envs/omnimergekit/bin/python \
  /srv/ml/repos/omnimergekit/scripts/gguf_probe_header.py "$OUT" 2>&1 \
  | grep -E 'general.name|eos_token_id|chat_template *=|file_type'

echo "[$(ts)] === STD27_Q4KM_DONE rc=0 ==="
