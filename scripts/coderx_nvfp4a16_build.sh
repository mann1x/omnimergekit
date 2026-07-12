#!/usr/bin/env bash
# coderx_nvfp4a16_build.sh — build NVFP4A16 (nvidia-modelopt) for the v7-coderx
# code4/lcb3 (CX16c4l3) re-release. LOCAL ONLY — NEVER uploads.
#
# Parity: ManniX-ITA/gemma-4-A4B-98e-v7-coderx-NVFP4A16 exists but holds the OLD
# fs2440 weights; this rebuilds it from code4/lcb3 bf16. Canonical recipe = the
# same quantize_any.py --method nvfp4a16 (default tatsu-lab/alpaca, 128 samples)
# that built 128e/v4/v5 NVFP4A16. modelopt MUST be 0.43.0 (0.44.0 has two Gemma 4
# regressions: config dict->list + bf16 calibration NaN — see memory).
#
# HARD GATE: upload to HF/ollama is a SEPARATE step, gated on card sign-off.
# GPU1 ONLY (user standing instruction: no GPU0 until told).
set -uo pipefail
SRC=/mnt/sdc/ml/cx_std16/CX16c4l3-bf16
DST=/mnt/sdc/ml/cx_std16/CX16c4l3-NVFP4A16
# dev-main modelopt (0.45.0.dev, _QuantFusedExperts present) — stable 0.43.0 dropped
# the fused-experts plugin and would silently emit BF16-sized garbage on Gemma 4 MoE.
# quantize_any.py selects the quant env via MODELOPT_ENV (default = stable, WRONG here).
export MODELOPT_ENV=/srv/ml/envs/envs/modelopt_dev
PY="$MODELOPT_ENV/bin/python"
QA=/srv/ml/repos/omnimergekit/scripts/quantize_any.py
LOG=/mnt/sdc/ml/coderx_nvfp4a16_build.log
LOCK=/mnt/sdc/ml/coderx_nvfp4a16_build.lock
GPU=1
ts(){ date '+%T %Z'; }

exec >>"$LOG" 2>&1
exec 9>"$LOCK"; flock -n 9 || { echo "[$(ts)] already running (lock held) — abort"; exit 0; }
echo "################ coderx NVFP4A16 build (LOCAL, no-upload) $(ts) ################"

# ---- preflight ----
[ -d "$SRC" ] || { echo "FATAL no src $SRC"; exit 2; }
"$PY" -c "from modelopt.torch.quantization.plugins.huggingface import _QuantFusedExperts" 2>/dev/null \
  || { echo "FATAL $MODELOPT_ENV missing _QuantFusedExperts (Gemma 4 MoE needs dev-main modelopt)"; exit 2; }
free=$(df -P /mnt/sdc | awk 'NR==2{print $4}'); [ "$free" -gt 60000000 ] || { echo "FATAL disk<60G ($free K)"; exit 2; }

echo "[$(ts)] MODELOPT_ENV=$MODELOPT_ENV CUDA_VISIBLE_DEVICES=$GPU quantize_any --method nvfp4a16 --src $SRC --dst $DST"
CUDA_VISIBLE_DEVICES=$GPU nice -n 10 "$PY" "$QA" --method nvfp4a16 --src "$SRC" --dst "$DST"
rc=$?; echo "[$(ts)] quantize_any exit=$rc"

# ── output canary: catch BF16-sized silent regression + NaN-calibration (0.44) ──
# bf16 baseline = real safetensors only (exclude .pre_shared_upweight backups in SRC).
bf16k=$(du -sck "$SRC"/*.safetensors 2>/dev/null | tail -1 | cut -f1)
dstk=$(du -sk "$DST" 2>/dev/null | cut -f1)
nshard=$(ls "$DST"/*.safetensors 2>/dev/null | wc -l)
frac=$(awk "BEGIN{printf \"%.3f\", ${dstk:-0}/${bf16k:-1}}")
echo "=== output canary ($(ts)) ==="
echo "  shards=$nshard size_frac=$frac (dst=${dstk}k bf16=${bf16k}k)"
canary_rc=0
{ [ "$nshard" -ge 1 ] && awk "BEGIN{exit !($frac>0.15 && $frac<0.70)}"; } || { echo "  FATAL: shards/size_frac (BF16-sized garbage or empty)"; canary_rc=8; }
if [ "$canary_rc" = 0 ]; then
  "$PY" - "$DST" <<'PYF'
import sys, glob, os, torch
from safetensors import safe_open
d=sys.argv[1]; bad=[]; nf=0
for f in sorted(glob.glob(os.path.join(d,"*.safetensors"))):
    with safe_open(f, framework="pt") as h:
        for k in h.keys():
            t=h.get_tensor(k)
            if t.is_floating_point():
                nf+=1
                if not torch.isfinite(t.float()).all(): bad.append(k)
print("  finite-check: %d float tensors, nonfinite=%d" % (nf, len(bad)))
sys.exit(0 if not bad else 9)
PYF
  canary_rc=$?
fi
[ -f "$DST/preprocessor_config.json" ] || echo "  NOTE: preprocessor_config.json absent (synth before vLLM serve)"
echo "  CANARY rc=$canary_rc (0=clean)"
echo "###### CODERX_NVFP4A16_DONE $(ts) rc=$rc canary=$canary_rc ######"
