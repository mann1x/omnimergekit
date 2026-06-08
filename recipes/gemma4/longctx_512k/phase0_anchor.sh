#!/bin/bash
# phase0_anchor.sh — solidpc anchor measurements (no blackswan-2 burn).
#
# For target T:
#   1. Canonical 9-bench at 32k context (regression bar)
#   2. RULER NIAH single-needle at 32k, 64k, 128k, 256k
#   3. MRCR-v2 8-needle at 128k
#   4. Coherence smoke @ 256k (one long-context prompt, manual inspection)
#
# Outputs land at:
#   backup_models/eval_results_longctx_512k_anchors/<target>/
#       9bench/                  ← omk_eval summary.json per bench
#       ruler_niah/{32k,64k,128k,256k}/
#       mrcr_v2/128k/
#       coherence_smoke_256k.md  ← human-readable transcript
#
# ### COUNCIL — read brief §"Phase 0" + plan §"Phase 0 — solidpc anchors".
# Open Q: which RULER fork do we use? lm-eval-harness has a `ruler/` tasks
# subdir as of 0.4.x — verify it covers single-needle and matches the
# `niah_single_*` length variants we'd add as omk templates.

set -uo pipefail

BM=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
OMK_PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK_EVAL=/shared/dev/omnimergekit/eval/omk_eval.py

TARGET=""
DO_RUN=0
SKIP_9BENCH=0
SKIP_NIAH=0
SKIP_MRCR=0
while [ $# -gt 0 ]; do
  case "$1" in
    --target)       TARGET=$2; shift 2;;
    --run)          DO_RUN=1; shift;;
    --skip-9bench)  SKIP_9BENCH=1; shift;;
    --skip-niah)    SKIP_NIAH=1; shift;;
    --skip-mrcr)    SKIP_MRCR=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

case "$TARGET" in
  31b) MODEL_DIR=$BM/google/gemma-4-31B-it
       SERVED=gemma-4-31B-it-q6k         # llama.cpp Q6_K serving
       GGUF=$BM/google/gemma-4-31B-it-GGUF/gemma-4-31B-it.Q6_K.gguf
       ;;
  v6c) MODEL_DIR=$BM/google/gemma-4-A4B-98e-v6-coder-it
       SERVED=gemma-4-A4B-98e-v6-coder-it-q6k
       GGUF=$BM/google/gemma-4-A4B-98e-v6-coder-it-GGUF/gemma-4-A4B-98e-v6-coder-it.Q6_K.gguf
       ;;
  *) echo "FATAL: --target must be 31b|v6c"; exit 2;;
esac

OUT=$BM/eval_results_longctx_512k_anchors/$TARGET
mkdir -p "$OUT/9bench" "$OUT/ruler_niah" "$OUT/mrcr_v2/128k"

echo "=== Phase 0 anchors: $TARGET ==="
echo "  model_dir : $MODEL_DIR"
echo "  gguf      : $GGUF"
echo "  out       : $OUT"
echo

if [ "$DO_RUN" -ne 1 ]; then
  echo "[dry-run] would launch:"
fi

# ---------- 1. 9-bench at 32k ----------
if [ "$SKIP_9BENCH" -ne 1 ]; then
  echo "[1/4] 9-bench at 32k (canonical greedy, EVAL_PROTOCOL stack@2)"
  if [ "$DO_RUN" -eq 1 ]; then
    # eval_suite_llama.sh handles 9-bench Q6_K on solidpc — re-use it.
    # TODO COUNCIL: confirm we should re-eval rather than use cached
    # results from T48/T49 (31B) / T92/T93 (v6-coder). Re-eval is the safe
    # default for the *anchor* role (pre-extension baseline).
    bash $BM/scripts/eval_suite_llama.sh \
      --gguf "$GGUF" \
      --name "$SERVED" \
      --output-root "$OUT/9bench" \
      --use-stack-pinned-vllm 0
  fi
fi

# ---------- 2. RULER NIAH single-needle at {32k, 64k, 128k, 256k} ----------
if [ "$SKIP_NIAH" -ne 1 ]; then
  for L in 32k 64k 128k 256k; do
    case "$L" in 32k) CTX=32768;; 64k) CTX=65536;; 128k) CTX=131072;; 256k) CTX=262144;; esac
    echo "[2/4] RULER NIAH single-needle @ $L (ctx=$CTX)"
    if [ "$DO_RUN" -eq 1 ]; then
      # TODO COUNCIL: which template? Need to add ruler_niah_single_<L>.yaml
      # to omk eval/templates/. Below assumes the template exists.
      "$OMK_PY" "$OMK_EVAL" \
        --template "ruler_niah_single_$L" \
        --model "$GGUF" \
        --served-name "$SERVED" \
        --results-dir "$OUT/ruler_niah/$L" \
        --backend llama \
        --ctx "$CTX"
    fi
  done
fi

# ---------- 3. MRCR-v2 8-needle @ 128k ----------
if [ "$SKIP_MRCR" -ne 1 ]; then
  echo "[3/4] MRCR-v2 8-needle @ 128k"
  if [ "$DO_RUN" -eq 1 ]; then
    "$OMK_PY" "$OMK_EVAL" \
      --template mrcr_v2_8needle_128k \
      --model "$GGUF" \
      --served-name "$SERVED" \
      --results-dir "$OUT/mrcr_v2/128k" \
      --backend llama \
      --ctx 131072
  fi
fi

# ---------- 4. Coherence smoke @ 256k (manual transcript) ----------
echo "[4/4] Coherence smoke @ 256k — human inspection required"
if [ "$DO_RUN" -eq 1 ]; then
  # Plan §Phase 0 step 4: launch llama-server at -c 262144, fire one
  # LongBench-v2 longest prompt, capture output, write to coherence_smoke_256k.md.
  # TODO: implement smoke prompt + post-it via curl, save raw transcript.
  echo "TODO: implement coherence smoke harness" > "$OUT/coherence_smoke_256k.md"
fi

echo
echo "=== Phase 0 anchors complete for $TARGET ==="
echo "    → archive these BEFORE Phase 1 launch (per feedback-eval-results-are-sacred)"
echo "    → read summary.json for 9-bench scores; raw lm_eval samples in lm_eval_out/"
