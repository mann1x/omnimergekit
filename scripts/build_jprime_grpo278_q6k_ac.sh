#!/usr/bin/env bash
# Q6_K + AtomicChat imatrix for Jprime-grpo-eff-278 — the ONLY basis that matches the
# existing base anchor at eval_results_llama_suite/jprime_p3_ac_q6k (Q6_K, AC imatrix
# 9438 chunks, entries 295, greedy). Building this reuses that anchor for free instead
# of re-evaluating the base at a different quant/calibration basis.
# The trained arm gets its OWN imatrix: reusing the base's would be invalid.
# --imatrix-chunks -1 == ALL chunks. The engine's default is 128, but the base anchor's
# AC imatrix consumed the WHOLE corpus (chunks_count 9438). A 128-chunk imatrix here
# would be a DIFFERENT calibration basis and the score comparison would be void.
set -euo pipefail
REPO=/srv/ml/repos/omnimergekit
PY=/srv/ml/envs/envs/omnimergekit/bin/python
NAME=Jprime-grpo-eff-278
MERGED=/mnt/sdc/v7rework/arms/${NAME}-bf16
SRC=/mnt/sdc/v7rework/quants/${NAME}-bf16
OUT=/mnt/sdc/v7rework/quants_ac/${NAME}-bf16
CAL=/mnt/sdc/ornith-pod/workspace_root/calib-ac/builds/gemma-4-26b-a4b/calib_train.txt

[ -f "$CAL" ] || { echo "FATAL: AC corpus missing: $CAL"; exit 1; }
[ -d "$MERGED" ] || { echo "FATAL: merged dir missing"; exit 1; }
AVAIL=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc '0-9')
[ "$AVAIL" -ge 80 ] || { echo "FATAL: ${AVAIL}G free, need >=80G"; exit 1; }
echo ">>> preflight ok: ${AVAIL}G free"

mkdir -p "$OUT"
# Hardlink the already-built F16 (same fs) so the 40GB conversion is skipped.
F16="${NAME}-bf16-F16.gguf"
if [ ! -f "$OUT/$F16" ] && [ -f "$SRC/$F16" ]; then
  ln "$SRC/$F16" "$OUT/$F16" && echo ">>> hardlinked existing F16 (conversion skipped)"
fi

echo ">>> $(date -u +%FT%TZ) Q6_K + AC imatrix (9438-chunk corpus) — expect ~80min imatrix"
cd "$REPO"
export OMK_NO_README=1
"$PY" scripts/quantize_gguf.py \
    --model "$MERGED" \
    --base-model-id ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it \
    --output-dir "$OUT" \
    --only Q6_K \
    --force-imatrix \
    --imatrix-chunks -1 \
    --cal-data "$CAL" \
    --no-upload --keep-local --ngl 99

echo ">>> $(date -u +%FT%TZ) Q6K_AC DONE"
ls -la "$OUT"
