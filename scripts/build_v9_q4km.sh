#!/usr/bin/env bash
# build_v9_q4km.sh — Q4_K_M for the v9-agentic student, reusing the SAME calib_v5 imatrix
# already computed for the Q6_K (a pruned variant cannot borrow a base imatrix).
# CPU-only (--ngl 0) so it coexists with the LCB eval holding GPU0.
# IMATRIX_EXCLUDE is retired (empty) since omk 0530295, so every _K tier is calibrated by
# default and --force-imatrix is a no-op; we assert the imatrix at GGUF-KV level anyway.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
WRAP=/srv/ml/scripts/qg_imatrix_all.py
SCR=/srv/ml/repos/omnimergekit/scripts
WORK=/mnt/sdc/ml/v9build
BF=$WORK/v9-agentic-bf16
GG=$WORK/v9-agentic-GGUF
TMPL=$WORK/tmpl/gemma-4-A4B-98e-v7-coder-it-Q6_K.jinja
TSHA=2e4c13d90aa29be968d48e4eecfa8da22e2656d890149e7924b445865490cbac
ts(){ date -u '+%H:%M:%SZ'; }

echo "===== V9 Q4_K_M START $(ts) ====="
[ -s "$GG/imatrix.dat" ] || { echo "FATAL: no imatrix.dat in $GG — refusing to build uncalibrated"; exit 2; }
echo "[q4km $(ts)] reusing imatrix: $(stat -c%s "$GG/imatrix.dat") bytes"
df -h "$GG" | tail -1

CUDA_VISIBLE_DEVICES="" "$PY" "$WRAP" \
  --model "$BF" --output-dir "$GG" --only Q4_K_M \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it \
  --no-upload --keep-local --ngl 0 --sanity-check \
  || { echo "FATAL quantize rc=$?"; exit 1; }

Q=$(ls "$GG"/*Q4_K_M*.gguf 2>/dev/null | head -1)
[ -n "$Q" ] || { echo "FATAL no Q4_K_M produced"; exit 1; }
echo "[q4km $(ts)] built: $Q ($(stat -c%s "$Q") bytes)"

# Assert the imatrix actually took, at the KV level -- not by trusting the flag.
"$PY" - "$Q" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
want = ["quantize.imatrix.file", "quantize.imatrix.dataset",
        "quantize.imatrix.entries_count", "quantize.imatrix.chunks_count"]
have = {f.name for f in r.fields.values()}
missing = [k for k in want if k not in have]
for k in want:
    if k in have:
        f = r.fields[k]
        try:
            v = f.parts[f.data[0]]
            v = bytes(v).decode("utf-8", "replace") if v.dtype.kind in "iu" and len(v) > 8 else v[0]
        except Exception:
            v = "<unreadable>"
        print("  %-32s = %s" % (k, v))
print("IMATRIX_KV_GATE: %s" % ("PASS" if not missing else "FAIL missing=%s" % missing))
sys.exit(0 if not missing else 1)
PY
echo "V9_Q4KM=$Q"
sha256sum "$Q" > "$Q.sha256"
echo "###### V9_Q4KM_DONE $(ts) ######"
