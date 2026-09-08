#!/usr/bin/env bash
# REBUILD Q4_K_M *WITH* imatrix. --force-imatrix overrides IMATRIX_EXCLUDE, which is a
# GEMMA-measured crossover (v7-coder/v6-coder ladders, per the constant's own comment).
# Qwen3.6-27B-A3B's crossover is UNMEASURED, and the shipped sibling + our own armJ Q6_K
# eval artifact are both imat (calibration_datav5, 510 entries / 128 chunks) -- so the
# release must match, not diverge. See feedback_imatrix_use_is_a_gguf_kv_fact.
# imatrix.dat already exists in the output dir, so this reuses it (no GPU recompute).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit
OUT=/mnt/sdc/ream-work/gguf_coderx
echo "[imat $(date -u +%H:%M:%SZ)] imatrix.dat present: $(ls -la $OUT/imatrix.dat 2>/dev/null | awk '{print $5}') bytes"
"$PY" "$OMK/scripts/quantize_gguf.py" \
    --model /mnt/sdc/ream-work/publish/Qwen3.6-27B-A3B-CoderX \
    --output-dir "$OUT" \
    --only Q4_K_M --force-imatrix \
    --no-upload --keep-local \
    --base-model-id Qwen/Qwen3.6-35B-A3B 2>&1 | tail -30
# THE FLAG PROVES INTENT; ONLY THE KV PROVES OUTCOME.
"$PY" - <<'PYEOF'
from gguf import GGUFReader
p = "/mnt/sdc/ream-work/gguf_coderx/Qwen3.6-27B-A3B-CoderX-Q4_K_M.gguf"
r = GGUFReader(p)
kv = {f.name: f.contents() for f in r.fields.values() if "imatrix" in f.name}
print("IMAT KV:", kv)
print("GATE:", "PASS" if kv.get("quantize.imatrix.file") else "*** FAIL - still noimat ***")
PYEOF
echo "CODERX_Q4KM_IMAT_DONE"
