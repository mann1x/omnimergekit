#!/usr/bin/env bash
# nvfp4_v8_noupload.sh — build v8 (fkbroad-soft2) NVFP4A16 to disk + HARD output
# canary. NO upload (publishing is gated). Mirrors nvfp4_v7.sh's canary exactly
# (shards>=1, size_frac in [0.15,0.70], all float tensors finite) but stops
# before the hf upload. Run on GPU1.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
export MODELOPT_ENV=/srv/ml/envs/envs/modelopt_dev
export HF_XET_HIGH_PERFORMANCE=1
OMK=/srv/ml/repos/omnimergekit
PY_OMK=/srv/ml/envs/envs/omnimergekit/bin/python
QANY=$OMK/scripts/quantize_any.py
VPY=/srv/ml/envs/envs/modelopt_dev/bin/python
BF16=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-it
DST=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-NVFP4A16-v8
L(){ echo "[nvfp4-v8 $(date -u +%H:%M:%S)] $*"; }

L "###### v8 NVFP4A16 build (GPU1, modelopt_dev, NO UPLOAD) ######"
[ -d "$BF16" ] || { L "FATAL $BF16 missing"; exit 1; }
bf16k=$(du -sk "$BF16" | cut -f1)
if [ "$(ls "$DST"/*.safetensors 2>/dev/null | wc -l)" -ge 1 ]; then
  L "reuse existing $DST (resumable)"
else
  rm -rf "$DST"
  L "quant $BF16 -> $DST"
  "$PY_OMK" "$QANY" --src "$BF16" --dst "$DST" --method nvfp4a16 \
    || { L "FATAL quantize_any failed"; exit 1; }
fi
nshard=$(ls "$DST"/*.safetensors 2>/dev/null | wc -l)
dstk=$(du -sk "$DST" | cut -f1)
frac=$(awk "BEGIN{printf \"%.3f\", $dstk/$bf16k}")
L "canary: shards=$nshard size_frac=$frac (dst=${dstk}k bf16=${bf16k}k)"
[ "$nshard" -ge 1 ] || { L "FATAL no safetensors"; exit 1; }
awk "BEGIN{exit !($frac>0.15 && $frac<0.70)}" || { L "FATAL size_frac $frac outside [0.15,0.70]"; exit 1; }
"$VPY" - "$DST" <<'PY' || { L "FATAL finite check failed"; exit 1; }
import sys, glob, os
from safetensors import safe_open
import torch
d = sys.argv[1]; bad = []; nt = 0; nf = 0
for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
    with safe_open(f, framework="pt") as h:
        for k in h.keys():
            t = h.get_tensor(k); nt += 1
            if t.is_floating_point():
                nf += 1
                if not torch.isfinite(t.float()).all():
                    bad.append(k)
print("  finite-check: %d tensors (%d float); nonfinite=%d" % (nt, nf, len(bad)))
if bad:
    print("  NONFINITE:", bad[:10]); sys.exit(1)
PY
[ -f "$DST/preprocessor_config.json" ] || cp "$BF16/preprocessor_config.json" "$DST/" 2>/dev/null || true
L "canary PASS — NVFP4A16 built + verified at $DST (NOT uploaded — gated)"
L "###### NVFP4_V8_DONE ######"
