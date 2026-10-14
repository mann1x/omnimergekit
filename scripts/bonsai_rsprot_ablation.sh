#!/usr/bin/env bash
# bonsai_rsprot_ablation.sh — R4: measure PrismML's ONE documented gen1->gen2 method change.
#
# THE CLAIM UNDER TEST. bonsai-2-27b-whitepaper §2.2: Bonsai 2 holds 26,238,464 params
# (0.0976%) above the low-bit representation -- "neither rotated nor quantized" -- "while the
# previous release ternarized these alongside everything else". Across that change PrismML's
# reported retention went 94.6% (gen-1 ternary 27B) -> 98.2% (gen 2). If protecting 0.0976% of
# the weights carries a large share of that, recurrent-state error accumulation is the binding
# constraint in hybrid-attention ternary models.
#
# WHAT MAKES THIS MEASURABLE FOR US AND NOBODY ELSE. We read PrismML's actual per-tensor type
# map out of their published file (851 tensors: 402 PTQ1_0 / 353 F32 / 96 BF16) and our own
# writer's map for the same architecture (851: 496 PQ2_0 / 353 F32 / 1 Q6_K / 1 Q4_K). The
# partition differs in EXACTLY the Table 2 families:
#   PrismML keeps blk.*.ssm_alpha.weight + blk.*.ssm_beta.weight (96 tensors = the paper's
#   in_proj_a/in_proj_b, 23.6M of their 26.2M preserved params) at BF16. We ternarize them.
# Every norm PrismML lists is ALREADY F32 in our output, so the ablation reduces to those 96
# tensors -- one flag on an otherwise byte-identical writer.
#
# BASIS DISCIPLINE -- WHY BOTH ARMS ARE RE-MEASURED HERE.
# The logged arm-A result (KLD 12.909, same-top-1 0.000%) was produced on a ggml-cuda.so of
# 314,877,984 bytes; the library in this engine dir is 314,877,792 bytes, rebuilt 2026-10-09
# 14:31 AFTER that run. The serving library's build is part of the basis (a v6 plain-vs-AC
# comparison straddling a rebuild manufactured a -5pp effect that did not exist). So this
# script re-runs BOTH arms back to back on ONE pinned binary. The logged 12.909 is kept only as
# a library-drift check, never as the comparison arm.
#
# EXPECTED FLOOR EFFECT, STATED UP FRONT. Arm A sits at same-top-1 = 0.000% because our files
# carry no prism.hadamard metadata and the runtime therefore applies NO rotation (raw basis in,
# raw basis out). Two floored arms would measure nothing. This run is still worth its minutes
# because the outcome is informative either way:
#   - arm B lifts off 0.000%  -> the recurrent path alone is a first-order term; big result.
#   - arm B also floors       -> the rotation dominates, R1 must precede R4, and this bounds
#                                the recurrent-path effect FROM ABOVE on an unrotated basis.
# It is NOT a test of PrismML's 94.6->98.2 claim, which lives in the rotated basis.
set -uo pipefail
ts(){ date -u '+%H:%M:%SZ'; }

QUANT=/srv/ml/repos/omnimergekit/tools/opencoti-quantize/bin/opencoti-quantize
PPL=/srv/ml/bonsai-assess/c11-33/side-cu/perplexity
CULIB=/srv/ml/bonsai-assess/c11-33/side-cu/ggml-cuda.so
DIR=/mnt/sdc/opencoti-models/sdc-ml/omnimerge-v6
SRC=$DIR/Qwen3.8-27B-Omnimerge-v6-BF16.gguf
ARM_A=$DIR/Qwen3.8-27B-Omnimerge-v6-PQ2_0.gguf
ARM_B=$DIR/Qwen3.8-27B-Omnimerge-v6-PQ2_0-rsprot.gguf
BASEKLD=/mnt/sdc/opencoti-scratch/v6-ternary/bf16-c2048.kld
TEXT=/srv/ml/ppl-corpus/wiki.test.mid.raw
OUT=/mnt/sdc/opencoti-scratch/v6-ternary
READER=/srv/ml/scripts/gguf_read_prism_meta.py
THREADS="${THREADS:-32}"

for f in "$QUANT" "$PPL" "$CULIB" "$SRC" "$ARM_A" "$BASEKLD" "$TEXT" "$READER"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

echo "===== BONSAI RSPROT ABLATION $(ts) ====="
echo "[basis] engine   : $PPL  $(stat -c%s "$PPL") B"
echo "[basis] cuda lib : $CULIB  $(stat -c%s "$CULIB") B  (arm-A log recorded 314877984)"
echo "[basis] quantizer: $(sha256sum "$QUANT" | cut -c1-16)"
echo "[basis] base kld : $BASEKLD  $(stat -c%s "$BASEKLD") B"
df -h /mnt/sdc | tail -1

# ── 1. arm B: identical writer, ssm_alpha/ssm_beta held at BF16 ───────────────
if [ -f "$ARM_B" ]; then
  echo "[1 $(ts)] arm B exists, skipping quantize"
else
  echo "[1 $(ts)] quantize arm B (PQ2_0, ssm_alpha+ssm_beta -> bf16)"
  /bin/sh "$QUANT" \
    --tensor-type ssm_alpha=bf16 \
    --tensor-type ssm_beta=bf16 \
    "$SRC" "$ARM_B" PQ2_0 "$THREADS" > "$OUT/quantize-PQ2_0-rsprot.log" 2>&1
  RC=$?
  if [ $RC -ne 0 ] || [ ! -s "$ARM_B" ]; then
    echo "[1 $(ts)] FATAL quantize rc=$RC"; tail -20 "$OUT/quantize-PQ2_0-rsprot.log"; exit 1
  fi
  sha256sum "$ARM_B" | tee "$ARM_B.sha256"
fi
echo "[1 $(ts)] arm B $(stat -c%s "$ARM_B") bytes"

# ── 2. the protection must actually be in the file ───────────────────────────
echo "[2 $(ts)] tensor census, both arms (asking for a type is not getting it)"
for L in "A:$ARM_A" "B:$ARM_B"; do
  N=${L%%:*}; F=${L#*:}
  echo -n "   arm $N  "; python3 "$READER" --tensors "$F" 2>&1 | head -1
done
python3 - "$READER" "$ARM_B" <<'PYEOF' || exit 3
import subprocess, sys, re
reader, arm_b = sys.argv[1], sys.argv[2]
out = subprocess.run([sys.executable, reader, "--tensors", arm_b],
                     capture_output=True, text=True).stdout
bad = [ln for ln in out.splitlines()
       if re.search(r"ssm_(alpha|beta)\.weight", ln) and "BF16" not in ln]
if bad:
    sys.exit("FATAL: %d ssm_alpha/beta tensors are NOT BF16, e.g. %s"
             % (len(bad), bad[0].strip()))
n = len([ln for ln in out.splitlines() if re.search(r"ssm_(alpha|beta)\.weight", ln)])
if n != 96:
    sys.exit("FATAL: expected 96 ssm_alpha/beta tensors, found %d" % n)
print("   arm B: all 96 ssm_alpha/ssm_beta tensors are BF16 — protection IS in the file")
PYEOF

# ── 3. KLD, both arms, same binary, back to back ─────────────────────────────
for L in "A:$ARM_A" "B:$ARM_B"; do
  N=${L%%:*}; F=${L#*:}
  LOG="$OUT/kld-rsprot-arm$N.log"
  echo "[3 $(ts)] KLD arm $N -> $LOG"
  CUDA_VISIBLE_DEVICES=0 "$PPL" -m "$F" -f "$TEXT" \
    --kl-divergence-base "$BASEKLD" --kl-divergence \
    -c 2048 --chunks 8 -b 2048 -ub 512 -ngl 99 > "$LOG" 2>&1
  RC=$?
  [ $RC -eq 0 ] || { echo "   FAIL rc=$RC"; tail -12 "$LOG"; continue; }
  echo "   --- arm $N ---"
  grep -E "^(Mean  *KLD|Minimum KLD|Mean  *Δp|RMS Δp|Same top p)" "$LOG" | sed 's/^/     /'
  grep -iE "^(Final estimate|PPL)" "$LOG" | sed 's/^/     /'
done

echo "[$(ts)] === RSPROT_ABLATION_DONE ==="
