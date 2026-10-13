#!/usr/bin/env bash
# build_v9_ac_quants.sh — v9-agentic Q6_K + Q4_K_M quantized with the ATOMICCHAT imatrix.
#
# This is the second arm of a pub-vs-AC pair. The calib_v5 quants
# (v9-agentic-bf16-Q6_K.gguf / -Q4_K_M.gguf) are the first arm and MUST survive, so the
# AC outputs carry an -AC suffix. Nothing here overwrites an existing quant or imatrix.
#
# Source is the F16, never an existing Q6 — re-quantizing a quant is double quantization.
set -uo pipefail
ts() { date -u +%H:%M:%SZ; }

LCPP=/mnt/sdc/ml/llama.cpp-b9700
GG=/mnt/sdc/ml/v9build/v9-agentic-GGUF
F16="$GG/v9-agentic-bf16-F16.gguf"
IMAT="$GG/imatrix_AC.dat"
THREADS="${THREADS:-32}"

for f in "$LCPP/build/bin/llama-quantize" "$F16" "$IMAT"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

# The imatrix must be the AC one. Assert identity, do not trust the path.
MAGIC=$(head -c 4 "$IMAT")
[ "$MAGIC" = "GGUF" ] || { echo "FATAL: imatrix is not GGUF-format (magic=$MAGIC)"; exit 3; }
echo "[$(ts)] imatrix: $(stat -c%s "$IMAT") B sha=$(sha256sum "$IMAT" | cut -c1-16)"
if ! /root/anaconda3/envs/omnimergekit/bin/python - "$IMAT" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
n = r.fields["imatrix.chunk_count"].parts[r.fields["imatrix.chunk_count"].data[0]].tolist()[0]
print("[imatrix] chunk_count=%d entries=%d" % (n, len(r.tensors)))
if n != 9585:
    sys.exit("FATAL: expected the 9585-chunk AtomicChat imatrix, got %d" % n)
PY
then
  exit 4
fi

df -h /mnt/sdc | tail -1

for TIER in Q6_K Q4_K_M; do
  OUT="$GG/v9-agentic-bf16-${TIER}-AC.gguf"
  if [ -f "$OUT" ]; then
    echo "[$(ts)] $TIER-AC exists, skipping"; continue
  fi
  echo "[$(ts)] quantize $TIER WITH the AtomicChat imatrix -> $(basename "$OUT")"
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$OUT" "$TIER" "$THREADS" 2>&1 | tail -4
  if [ ! -s "$OUT" ]; then echo "[$(ts)] FATAL: $TIER produced nothing"; exit 5; fi
  echo "[$(ts)] $TIER-AC $(stat -c%s "$OUT") bytes"
  sha256sum "$OUT" | tee "$OUT.sha256"

  # Prove the imatrix rode into the file — a GGUF KV fact, not a flag we passed.
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
    sys.exit("FATAL: no imatrix KV in the shipped file — it was quantized WITHOUT it")
for k in sorted(hits):
    print("   %-38s %s" % (k, str(hits[k])[:110]))
PY
  then
    exit 6
  fi
done

echo "[$(ts)] calib_v5 arm still present:"
ls -l "$GG"/v9-agentic-bf16-Q6_K.gguf "$GG"/v9-agentic-bf16-Q4_K_M.gguf 2>/dev/null
echo "[$(ts)] === V9_AC_QUANTS_DONE rc=0 ==="
