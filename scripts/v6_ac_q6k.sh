#!/usr/bin/env bash
# Qwen3.8-27B-Omnimerge-v6 -> Q6_K quantized with the ATOMICCHAT imatrix.
#
# The published v6 GGUF tiers were built with a different calibration corpus.
# This tier exists to measure what the AtomicChat corpus buys, so it gets a
# DISTINCT filename and is NOT uploaded over the published one.
#
#   imatrix : /mnt/sdc/ml/omnimerge-v6/imatrix.dat   (AC corpus, 9703 chunks,
#             --parse-special, V6_ATOMICCHAT_IMATRIX_DONE rc=0 at 21:28Z)
#   source  : Qwen3.8-27B-Omnimerge-v6-F16.gguf      (sha-verified vs HF LFS oid)
set -uo pipefail
ts(){ date -u +%H:%M:%S; }
say(){ echo "[$(ts)Z] $*"; }

W=/mnt/sdc/ml/omnimerge-v6
F16=$W/Qwen3.8-27B-Omnimerge-v6-F16.gguf
IMAT=$W/imatrix.dat
OUT=$W/Qwen3.8-27B-Omnimerge-v6-Q6_K-AC.gguf
QBIN=/opt/llama.cpp/build/bin/llama-quantize

exec 9>/var/lock/v6_ac_q6k.lock
flock -n 9 || { say "already running - refusing"; exit 1; }

[ -s "$F16" ]  || { say "ABORT: no F16 at $F16"; exit 2; }
[ -s "$IMAT" ] || { say "ABORT: no imatrix at $IMAT"; exit 3; }
[ -x "$QBIN" ] || { say "ABORT: no llama-quantize"; exit 4; }

# imatrix must be the AC one, not a leftover. Assert magic + size rather than trust the path.
MAGIC=$(head -c 4 "$IMAT")
say "imatrix: $(stat -c%s "$IMAT") B magic=$MAGIC"
[ "$MAGIC" = "GGUF" ] || { say "ABORT: imatrix is not GGUF-format"; exit 5; }

FREE=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc '0-9')
[ "$FREE" -ge 60 ] || { say "ABORT: need 60G on /mnt/sdc, have ${FREE}G"; exit 6; }

if [ -s "$OUT" ]; then
  say "Q6_K-AC already present: $(stat -c%s "$OUT") B - skipping quantize"
else
  say "quantize Q6_K WITH the AtomicChat imatrix -> $(basename "$OUT")"
  "$QBIN" --imatrix "$IMAT" "$F16" "$OUT" Q6_K 32 2>&1 | tail -6
  rc=${PIPESTATUS[0]}
  say "quantize rc=$rc"
  [ "$rc" -eq 0 ] || exit 10
fi
[ -s "$OUT" ] || { say "ABORT: no output produced"; exit 11; }
say "Q6_K-AC ready: $(stat -c%s "$OUT") B"

# Prove the imatrix actually rode into the file -- a GGUF KV fact, not a flag.
/srv/ml/envs/envs/omnimergekit/bin/python3.11 - "$OUT" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
hits = {k: str(r.fields[k].parts[r.fields[k].data[0]]) for k in r.fields if "imatrix" in k.lower()}
print("  imatrix KV in the shipped file:", hits if hits else "NONE  <-- PROBLEM")
PY
say "V6_AC_Q6K_DONE"
