#!/usr/bin/env bash
# nvfp4_std16.sh — NVFP4A16 quant + upload for STD16 (the no-DERN v7-coder release).
# Single model: STD16-bf16 -> ManniX-ITA/gemma-4-A4B-98e-v7-coder-NVFP4A16 (OVERWRITES old g15f2440).
# Byte-faithful to nvfp4_v7.sh: modelopt_dev (0.45dev, _QuantFusedExperts for the MoE), GPU0,
# hard output canary (shards>=1, size_frac in [0.15,0.70], all-finite) BEFORE upload.
# Launch:  source ~/.bashrc; setsid nohup bash nvfp4_std16.sh >LOG 2>&1 </dev/null &
set -uo pipefail
# NVFP4 export packs FP4 weights and needs a near-full GPU; the export OOM'd at 08:39 sharing
# GPU0 with 3 eval workers (only 2 MiB free). Wait for a free GPU below + expandable segments.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODELOPT_ENV=/srv/ml/envs/envs/modelopt_dev
export HF_XET_HIGH_PERFORMANCE=1
: "${HF_TOKEN:?HF_TOKEN must be exported (source ~/.bashrc before launch)}"
GOOGLE=/mnt/sdc/ml/google
OMK=/srv/ml/repos/omnimergekit
PY_OMK=/srv/ml/envs/envs/omnimergekit/bin/python
HF=/srv/ml/envs/envs/omnimergekit/bin/hf
QANY=$OMK/scripts/quantize_any.py
VPY=/srv/ml/envs/envs/modelopt_dev/bin/python   # torch+safetensors for the finite check
BF16=/mnt/sdc/ml/t223_fk/STD16-bf16
# Distinct STD16 build dir — must NOT collide with the pre-existing g15f2440 dir
# ($GOOGLE/gemma-4-A4B-98e-v7-coder-NVFP4A16), whose "reuse existing" guard would
# otherwise skip the re-quant and upload the WRONG (old) model. Upload still targets
# the same REPO, overwriting g15f2440 on HF with STD16.
DST=$GOOGLE/gemma-4-A4B-98e-v7-coder-STD16-NVFP4A16
REPO=ManniX-ITA/gemma-4-A4B-98e-v7-coder-NVFP4A16
L(){ echo "[nvfp4-std16 $(date -u +%H:%M:%S)] $*"; }

L "================= NVFP4A16 STD16 -> v7-coder ================="
[ -d "$BF16" ] || { L "FATAL: $BF16 missing"; exit 1; }
[ -f "$BF16/config.json" ] || { L "FATAL: $BF16/config.json missing"; exit 1; }
bf16k=$(du -sk "$BF16" | cut -f1)

# Wait for a GPU with >= 85 GB free (the eval holds both GPUs; export needs near-full GPU).
MINFREE_MIB=87000
GPU=""
for _ in $(seq 1 240); do   # 240 * 60s = up to 4h
  while IFS=, read -r idx fmib; do
    idx=$(echo "$idx" | tr -dc 0-9); fmib=$(echo "$fmib" | tr -dc 0-9)
    if [ "${fmib:-0}" -ge "$MINFREE_MIB" ]; then GPU="$idx"; break; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null)
  [ -n "$GPU" ] && break
  L "waiting for a GPU with >= ${MINFREE_MIB}MiB free (eval still running)..."
  sleep 60
done
[ -n "$GPU" ] || { L "FATAL: no GPU with >= ${MINFREE_MIB}MiB free after 4h wait"; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU"
L "using free GPU $GPU (export needs near-full GPU; expandable_segments on)"
if [ "$(ls "$DST"/*.safetensors 2>/dev/null | wc -l)" -ge 1 ]; then
  L "reuse existing $DST (skip re-quant; resumable)"
else
  rm -rf "$DST"
  L "quant $BF16 -> $DST (modelopt_dev)"
  "$PY_OMK" "$QANY" --src "$BF16" --dst "$DST" --method nvfp4a16 \
      || { L "FATAL: quantize_any failed"; exit 1; }
fi

# ── output canary ──────────────────────────────────────────────
nshard=$(ls "$DST"/*.safetensors 2>/dev/null | wc -l)
dstk=$(du -sk "$DST" | cut -f1)
frac=$(awk "BEGIN{printf \"%.3f\", $dstk/$bf16k}")
L "canary: shards=$nshard size_frac=$frac (dst=${dstk}k bf16=${bf16k}k)"
[ "$nshard" -ge 1 ] || { L "FATAL: no safetensors in $DST"; exit 1; }
awk "BEGIN{exit !($frac>0.15 && $frac<0.70)}" \
    || { L "FATAL: size_frac $frac outside [0.15,0.70] — BF16-sized garbage or empty"; exit 1; }
"$VPY" - "$DST" <<'PY' || { L "FATAL: finite check failed"; exit 1; }
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
L "canary PASS"

# ── preprocessor + upload ──────────────────────────────────────
[ -f "$DST/preprocessor_config.json" ] || cp "$BF16/preprocessor_config.json" "$DST/" 2>/dev/null || true
L "upload -> $REPO"
"$HF" upload "$REPO" "$DST" . \
    --exclude ".shared_applied" --exclude "*.pre_shared" \
    --commit-message "STD16 (no-DERN v7-coder) NVFP4A16 (modelopt 0.45dev _QuantFusedExperts)" \
    || { L "FATAL: upload failed"; exit 1; }
L "================= DONE STD16 NVFP4A16  NVFP4_STD16_DONE ================="
