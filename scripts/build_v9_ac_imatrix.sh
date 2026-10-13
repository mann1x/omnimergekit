#!/usr/bin/env bash
# build_v9_ac_imatrix.sh — AtomicChat imatrix for v9-agentic (Gemma-4 26B-A4B, 98e).
#
# WHY THIS EXISTS. v9-agentic already carries a calib_v5 imatrix (128 chunks,
# 44,263,808 B). This builds the SECOND arm — the AtomicChat corpus — so the two can
# be A/B'd the way v6 pub-vs-AC was. It does NOT replace the first.
#
# There is no published AtomicChat build for any Gemma model and none of the six
# existing builds is reusable: every AC/DSM marker is ABSENT from the Gemma-4 GGUF
# vocab (read out of tokenizer.ggml.tokens, not inferred from a manifest sha). The
# corpus consumed here was rendered fresh from the AC pool through the template
# SHIPPED inside the v9 GGUF — see recipes/gemma-4-26b-a4b.yaml in the
# AtomicChat/calib-corpora clone.
#
# TWO THINGS THIS SCRIPT REFUSES TO GET WRONG
#   1. It NEVER writes to imatrix.dat. The calib_v5 imatrix lives there and an
#      imatrix is never deleted or overwritten. Output is imatrix_AC.dat.
#   2. --parse-special is mandatory. llama-imatrix defaults it OFF, and without it
#      the 32,507 Gemma-4 control-token occurrences in the corpus tokenise as
#      literal punctuation — the agentic and reasoning slices would then calibrate
#      on text this model is never served. That is the entire point of this arm.
#
# No --chunks: the full corpus is consumed. Capping it silently discards the tail.
set -uo pipefail
ts() { date -u +%H:%M:%SZ; }

GPU="${1:-0}"
LCPP=/mnt/sdc/ml/llama.cpp-b9700
GG=/mnt/sdc/ml/v9build/v9-agentic-GGUF
F16="$GG/v9-agentic-bf16-F16.gguf"
CALIB=/mnt/sdc/ml/calib/ac_gemma4/calib_train.txt
OUT="$GG/imatrix_AC.dat"
LOG="$GG/imatrix_AC_run.log"

echo "[$(ts)] v9-agentic AtomicChat imatrix, GPU$GPU"

for f in "$LCPP/build/bin/llama-imatrix" "$F16" "$CALIB"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

# The existing calib_v5 imatrix must survive this run untouched.
PREV="$GG/imatrix.dat"
if [ -f "$PREV" ]; then
  PREV_SHA=$(sha256sum "$PREV" | cut -d' ' -f1)
  echo "[$(ts)] calib_v5 imatrix present: $(stat -c%s "$PREV") B sha=${PREV_SHA:0:16}"
fi
[ -f "$OUT" ] && { echo "FATAL: $OUT already exists — refusing to clobber an imatrix"; exit 3; }

# Corpus must actually carry Gemma-4 markup, or --parse-special buys nothing.
N_TURN=$(grep -aoF '<|turn>' "$CALIB" | wc -l)
N_TOOL=$(grep -aoF '<|tool_call>' "$CALIB" | wc -l)
echo "[$(ts)] corpus: $(stat -c%s "$CALIB") B  <|turn>=$N_TURN  <|tool_call>=$N_TOOL"
[ "$N_TURN" -gt 1000 ] || { echo "FATAL: corpus has no Gemma-4 turn markup ($N_TURN)"; exit 4; }

# Any foreign markup means the wrong corpus was staged.
for m in '<|im_start|>' '<think>' '<|DSM'; do
  n=$(grep -aoF "$m" "$CALIB" | wc -l)
  [ "$n" -eq 0 ] || { echo "FATAL: foreign markup $m x$n in corpus"; exit 5; }
done

df -h /mnt/sdc | tail -1

echo "[$(ts)] llama-imatrix --parse-special -ngl 99 (full corpus, no --chunks)"
CUDA_VISIBLE_DEVICES="$GPU" "$LCPP/build/bin/llama-imatrix" \
  -m "$F16" -f "$CALIB" -o "$OUT" --parse-special -ngl 99 \
  > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ] || [ ! -s "$OUT" ]; then
  echo "[$(ts)] IMATRIX_FAIL rc=$RC"; tail -25 "$LOG"; exit 1
fi

echo "[$(ts)] imatrix_AC.dat $(stat -c%s "$OUT") bytes (PRESERVED)"
sha256sum "$OUT"
grep -aoE "compute_imatrix: [0-9]+ chunks|n_chunk = [0-9]+|save_imatrix: stored collected data" "$LOG" | tail -5

# The calib_v5 imatrix must be byte-identical to what it was before this run.
if [ -f "$PREV" ]; then
  NOW_SHA=$(sha256sum "$PREV" | cut -d' ' -f1)
  if [ "$NOW_SHA" = "$PREV_SHA" ]; then
    echo "[$(ts)] calib_v5 imatrix UNTOUCHED (sha matches)"
  else
    echo "[$(ts)] FATAL: calib_v5 imatrix CHANGED"
    exit 6
  fi
fi
echo "[$(ts)] === V9_AC_IMATRIX_DONE rc=0 ==="
