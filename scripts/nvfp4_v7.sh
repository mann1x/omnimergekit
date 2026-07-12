#!/usr/bin/env bash
# NVFP4A16 quant + upload for the v7-coder cohort (both models) on bs2 GPU0.
# Uses the dev-main modelopt env (0.45.0.dev161, _QuantFusedExperts present) via
# the MODELOPT_ENV override added to quantize_any.py (omk commit 4999414).
# HARD output canary BEFORE upload: shards>=1, size_frac in [0.15,0.70] (catches
# the BF16-sized silent-regression), all float tensors finite (catches the 0.44
# NaN-calibration regression). v7-coder first; ABORT before v7-coderx if model 1
# fails the gate (dev-build trust gate — don't burn model 2 on garbage).
# Launch:  source ~/.bashrc; setsid nohup bash nvfp4_v7.sh >LOG 2>&1 </dev/null &
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
export MODELOPT_ENV=/srv/ml/envs/envs/modelopt_dev
export HF_XET_HIGH_PERFORMANCE=1
: "${HF_TOKEN:?HF_TOKEN must be exported (source ~/.bashrc before launch)}"
GOOGLE=/mnt/sdc/ml/google
OMK=/srv/ml/repos/omnimergekit
PY_OMK=/srv/ml/envs/envs/omnimergekit/bin/python
HF=/srv/ml/envs/envs/omnimergekit/bin/hf
QANY=$OMK/scripts/quantize_any.py
VPY=/srv/ml/envs/envs/modelopt_dev/bin/python   # has torch+safetensors for the finite check
L(){ echo "[nvfp4 $(date -u +%H:%M:%S)] $*"; }

# rows:  <local_bf16_basename>|<public_suffix>
ROWS=(
  "gemma-4-A4B-98e-v7-coder-g15f2440-it|v7-coder"
  "gemma-4-A4B-98e-v7-coder-fs2440-it|v7-coderx"
)

do_one(){
  local BF16BASE="$1" PUB="$2"
  local BF16="$GOOGLE/$BF16BASE"
  local DST="$GOOGLE/gemma-4-A4B-98e-${PUB}-NVFP4A16"
  local REPO="ManniX-ITA/gemma-4-A4B-98e-${PUB}-NVFP4A16"
  L "================= NVFP4A16 $PUB ================="
  [ -d "$BF16" ] || { L "FATAL: $BF16 missing"; return 1; }
  local bf16k; bf16k=$(du -sk "$BF16" | cut -f1)
  if [ "$(ls "$DST"/*.safetensors 2>/dev/null | wc -l)" -ge 1 ]; then
    L "[$PUB] reuse existing $DST (skip re-quant; resumable)"
  else
    rm -rf "$DST"
    L "[$PUB] quant $BF16 -> $DST"
    "$PY_OMK" "$QANY" --src "$BF16" --dst "$DST" --method nvfp4a16 \
        || { L "FATAL: quantize_any failed for $PUB"; return 1; }
  fi

  # ── output canary ──────────────────────────────────────────────
  local nshard dstk frac
  nshard=$(ls "$DST"/*.safetensors 2>/dev/null | wc -l)
  dstk=$(du -sk "$DST" | cut -f1)
  frac=$(awk "BEGIN{printf \"%.3f\", $dstk/$bf16k}")
  L "[$PUB] canary: shards=$nshard size_frac=$frac (dst=${dstk}k bf16=${bf16k}k)"
  [ "$nshard" -ge 1 ] || { L "FATAL: no safetensors in $DST"; return 1; }
  awk "BEGIN{exit !($frac>0.15 && $frac<0.70)}" \
      || { L "FATAL: size_frac $frac outside [0.15,0.70] — BF16-sized garbage or empty"; return 1; }
  "$VPY" - "$DST" <<'PY' || { L "FATAL: finite check failed for $PUB"; return 1; }
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
                # upcast to fp32: torch.isfinite is NotImplemented for Float8_e4m3fn
                # (NVFP4A16 stores block scales as FP8) — .float() handles all float dtypes.
                if not torch.isfinite(t.float()).all():
                    bad.append(k)
print("  finite-check: %d tensors (%d float); nonfinite=%d" % (nt, nf, len(bad)))
if bad:
    print("  NONFINITE:", bad[:10]); sys.exit(1)
PY
  L "[$PUB] canary PASS"

  # ── preprocessor + upload ──────────────────────────────────────
  [ -f "$DST/preprocessor_config.json" ] || cp "$BF16/preprocessor_config.json" "$DST/" 2>/dev/null || true
  L "[$PUB] upload -> $REPO"
  "$HF" upload "$REPO" "$DST" . \
      --exclude ".shared_applied" --exclude "*.pre_shared" \
      --commit-message "v7 cohort: ${PUB} NVFP4A16 (modelopt 0.45dev _QuantFusedExperts)" \
      || { L "FATAL: upload failed for $PUB"; return 1; }
  rm -f "$GOOGLE/NEEDS_NVFP4A16_${PUB}"
  L "================= DONE $PUB ================="
}

L "###### v7 NVFP4A16 START (GPU0, modelopt_dev) ######"
RC=0
for r in "${ROWS[@]}"; do
  IFS='|' read -r base pub <<<"$r"
  if [ -n "${ONLY:-}" ] && [ "$pub" != "$ONLY" ]; then L "skip $pub (ONLY=$ONLY)"; continue; fi
  if ! do_one "$base" "$pub"; then L "###### ABORT at $pub (model-gate) ######"; RC=1; break; fi
done
L "###### v7 NVFP4A16 END rc=$RC  NVFP4_ALL_DONE ######"
exit $RC
