#!/usr/bin/env bash
# CoderX full tier campaign -- 19 tiers at PARITY with the published sibling
# ManniX-ITA/Qwen3.6-27B-A3B-Coder-MTP-GGUF (user: "same tier list").
#
# --force-imatrix IS MANDATORY HERE (bug-618). quantize_gguf.py's IMATRIX_EXCLUDE
# {Q4_K_S/M/L, Q5_K_S/M/L, Q6_K, Q6_K_L} encodes a crossover measured on GEMMA
# families only (v7-coder 98e, v6-coder) -- the constant's own comment says so.
# The shipped Qwen sibling's KV is the authority on the Qwen recipe:
#   Q4_K_M / Q5_K_M / Q6_K / IQ4_XS all carry
#     quantize.imatrix.dataset = calibration_datav5.txt, 510 entries, 128 chunks
#   and only Q8_0 has none (correct -- Q8_0 is not an imatrix tier by rule).
#
# DISK: /mnt/sdc has ~279G free; the 19 tiers total ~300G. So NO --keep-local --
# each tier uploads then frees. The worker never deletes the base F16 or imatrix.dat,
# and local-dir source weights are explicitly not deleted (they are hardlinks to armJ).
#
# VISIBILITY: the target repo was pre-created PRIVATE. quantize_gguf.py calls
# create_repo(exist_ok=True) with no private= arg, which CANNOT flip an existing
# repo's visibility -- so nothing goes outward until the card is reviewed.
#
# NO --ollama-target: ollama tags need RENDERER/PARSER qwen3.5 + draft_num_predict,
# none of which this script can emit (--ollama-template has no qwen3.5 choice).
# The ollama leg is a separate publish shell.
#
# bs2: GPU1 only (GPU0 is not ours).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit
SRC=/mnt/sdc/ream-work/publish/Qwen3.6-27B-A3B-CoderX
OUT=/mnt/sdc/ream-work/gguf_coderx
REPO=ManniX-ITA/Qwen3.6-27B-A3B-CoderX-MTP-GGUF

# High tiers first: they are the ones users pull, and a disk stall later leaves the
# most valuable tiers already uploaded.
TIERS="Q8_0,Q6_K_L,Q6_K,Q5_K_L,Q5_K_M,Q5_K_S,Q4_K_L,Q4_K_M,Q4_K_S,IQ4_NL,IQ4_XS,Q3_K_XL,Q3_K_L,Q3_K_M,Q3_K_S,IQ3_M,Q2_K_L,IQ2_M,IQ2_XS"

say(){ echo "[campaign $(date -u +%H:%M:%SZ)] $*"; }

[ -d "$SRC" ] || { say "REFUSE: no source dir $SRC"; exit 1; }
# Recomputing the imatrix would silently change the recipe fingerprint mid-campaign.
[ -f "$OUT/imatrix.dat" ] || { say "REFUSE: no imatrix.dat at $OUT -- would recompute"; exit 1; }
free=$(df -BG --output=avail /mnt/sdc | tail -1 | tr -dc 0-9)
[ "${free:-0}" -ge 60 ] || { say "REFUSE: /mnt/sdc only ${free}G free"; exit 1; }

say "start: ${free}G free | imatrix $(stat -c%s "$OUT/imatrix.dat") B | repo $REPO (private)"

"$PY" "$OMK/scripts/quantize_gguf.py" \
    --model "$SRC" --output-dir "$OUT" --repo "$REPO" \
    --only "$TIERS" --force-imatrix \
    --base-model-id Qwen/Qwen3.6-35B-A3B 2>&1
rc=$?

say "quantize_gguf rc=$rc"
say "remaining locally:"; ls -la "$OUT"
echo "CODERX_CAMPAIGN_DONE rc=$rc"
