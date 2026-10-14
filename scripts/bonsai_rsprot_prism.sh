#!/usr/bin/env bash
# bonsai_rsprot_prism.sh — re-run R4 on the PRISM FORK, the reference engine.
#
# WHY THIS SUPERSEDES bonsai_rsprot_ablation.sh. That run used the opencoti engine, whose PQ
# kernels are c11 rows 3.2/3.3 and still in progress. An immature kernel cannot be told apart
# from a destroyed quantization by looking at the score: both read as collapse. So the earlier
# absolute numbers (and the conclusion drawn from them, that the ROTATION is the first-order
# term) were not safe. The reference engine is PrismML's own fork.
#
# Engine: llama-prism b10754 (commit 2459f68b5) — the exact build our opencoti-quantize writer
# was proven byte-identical against on LFM2.5-1.2B, so the writer and the reader agree by
# construction.
#
# THREE THINGS THIS DOES, IN ORDER
#  0. ENGINE CONTROL. PPL of PrismML's OWN published PQ2_0 on the prism fork AND on opencoti.
#     If the two agree, opencoti's kernels were trustworthy and the earlier run stands. If they
#     disagree, opencoti was the defect and every opencoti-measured ternary cell is void. This
#     needs no teacher logits, which is why it is the control and not the measurement.
#  1. BASE LOGITS ON THE PRISM FORK. The existing bf16-c2048.kld was written by opencoti;
#     using it here would mix engines inside one KLD. Written to a NEW name — an existing
#     measurement artifact is never overwritten.
#  2. KLD arm A (all ternary) vs arm B (96 ssm_alpha/ssm_beta at BF16), both on the fork,
#     against the fork's own base. Same two files as before; only the engine changes.
set -uo pipefail
ts(){ date -u '+%H:%M:%SZ'; }

PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68
PPL="$PRISM/llama-perplexity"
OC_PPL=/srv/ml/bonsai-assess/c11-33/side-cu/perplexity
DIR=/mnt/sdc/opencoti-models/sdc-ml/omnimerge-v6
BF16=$DIR/Qwen3.8-27B-Omnimerge-v6-BF16.gguf
ARM_A=$DIR/Qwen3.8-27B-Omnimerge-v6-PQ2_0.gguf
ARM_B=$DIR/Qwen3.8-27B-Omnimerge-v6-PQ2_0-rsprot.gguf
REF=/mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf
TEXT=/srv/ml/ppl-corpus/wiki.test.mid.raw
OUT=/mnt/sdc/opencoti-scratch/v6-ternary
BASE=$OUT/bf16-c2048-prism.kld
export LD_LIBRARY_PATH="$PRISM:${LD_LIBRARY_PATH:-}"

for f in "$PPL" "$OC_PPL" "$BF16" "$ARM_A" "$ARM_B" "$REF" "$TEXT"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
echo "===== RSPROT ON THE PRISM FORK $(ts) ====="
echo "[basis] prism    : $("$PPL" --version 2>&1 | head -1)"
echo "[basis] opencoti : $OC_PPL $(stat -c%s "$OC_PPL") B"
df -h /mnt/sdc | tail -1

show(){ grep -E "^(Final estimate|Mean  *KLD|Minimum KLD|Mean  *Δp|RMS Δp|Same top p)" "$1" | sed 's/^/     /'; }

# ── 0. ENGINE CONTROL on PrismML's own file ──────────────────────────────────
echo "[0 $(ts)] ENGINE CONTROL — PPL of PrismML's published PQ2_0, both engines"
for E in "prism:$PPL" "opencoti:$OC_PPL"; do
  N=${E%%:*}; B=${E#*:}
  L="$OUT/ctl-ref-prismfile-$N.log"
  echo "   [$N] -> $L"
  CUDA_VISIBLE_DEVICES=0 "$B" -m "$REF" -f "$TEXT" -c 2048 --chunks 8 -b 2048 -ub 512 -ngl 99 \
    > "$L" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then echo "     rc=$RC"; tail -6 "$L" | sed 's/^/     /'; else show "$L"; fi
done

# ── 1. base logits ON THE FORK ───────────────────────────────────────────────
if [ -s "$BASE" ]; then
  echo "[1 $(ts)] prism base logits exist ($(stat -c%s "$BASE") B), reusing"
else
  echo "[1 $(ts)] writing prism base logits -> $BASE"
  CUDA_VISIBLE_DEVICES=0 "$PPL" -m "$BF16" -f "$TEXT" --kl-divergence-base "$BASE" \
    -c 2048 --chunks 8 -b 2048 -ub 512 -ngl 99 > "$OUT/base-prism.log" 2>&1
  RC=$?
  [ $RC -eq 0 ] && [ -s "$BASE" ] || { echo "FATAL base logits rc=$RC"; tail -15 "$OUT/base-prism.log"; exit 1; }
  echo "[1 $(ts)] base $(stat -c%s "$BASE") B"
fi

# ── 2. the two arms, on the fork, against the fork's base ────────────────────
for L in "A:$ARM_A" "B:$ARM_B"; do
  N=${L%%:*}; F=${L#*:}
  LOG="$OUT/kld-prism-arm$N.log"
  echo "[2 $(ts)] KLD arm $N (prism fork) -> $LOG"
  CUDA_VISIBLE_DEVICES=0 "$PPL" -m "$F" -f "$TEXT" \
    --kl-divergence-base "$BASE" --kl-divergence \
    -c 2048 --chunks 8 -b 2048 -ub 512 -ngl 99 > "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then echo "   FAIL rc=$RC"; tail -10 "$LOG" | sed 's/^/     /'; continue; fi
  echo "   --- arm $N (prism) ---"; show "$LOG"
done

echo "[$(ts)] === RSPROT_PRISM_DONE ==="
