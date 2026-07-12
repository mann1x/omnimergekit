#!/usr/bin/env bash
# republish_noimat_flipped.sh — rebuild the high-bit K-quant tiers as NOIMAT and
# republish to HF + ollama for both v7 models, per the 2026-06-07 imatrix bit-depth
# CROSSOVER study. quantize_gguf.py IMATRIX_EXCLUDE now covers the whole Q4+ band
# (Q4_K_S/M/L, Q5_K_S/M/L, Q6_K[_L]) -> those tiers build imatrix-free automatically.
# Q4_K_M was ALREADY noimat on HF (prior single-tier exception), so the FLIPPED
# (rebuild) set is the OTHER 7 tiers below.
#
# CPU-only (--ngl 0): never contends with the GPU quant sweep. OMK_NO_README=1 keeps
# the curated model cards. CRITICAL — output filenames must match the PUBLISHED clean
# stems (-coder-it- / -coderx-it-), but get_model_name() derives the stem from the
# --model path BASENAME (quantize_gguf.py:497 + out_name :683). The local bf16 dirs
# are codenamed (-g15f2440- / -fs2440-), so we --model a CLEAN-NAMED SYMLINK; the
# existing -GGUF dirs already hold the matching clean-named F16 (reused, no reconvert,
# no wrong-name upload). imatrix.dat is untouched (still needed by Q3/Q2/IQ/CD tiers).
#
# Launch: source ~/.bashrc; setsid nohup bash republish_noimat_flipped.sh >LOG 2>&1 </dev/null &
set -uo pipefail
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
: "${HF_TOKEN:?HF_TOKEN must be exported (source ~/.bashrc before launch)}"
PY=/srv/ml/envs/envs/omnimergekit/bin/python
QGGUF=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
GOOGLE=/mnt/sdc/ml/google
export OMK_NO_README=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1

ONLY="Q4_K_S,Q4_K_L,Q5_K_S,Q5_K_M,Q5_K_L,Q6_K,Q6_K_L"   # Q4_K_M already noimat on HF

# clean_name | bf16dir | pub_suffix
ROWS=(
  "gemma-4-A4B-98e-v7-coder-it|gemma-4-A4B-98e-v7-coder-g15f2440-it|v7-coder"
  "gemma-4-A4B-98e-v7-coderx-it|gemma-4-A4B-98e-v7-coder-fs2440-it|v7-coderx"
)
L(){ echo "[republish $(date -u +%H:%M:%S)] $*"; }
L "START noimat republish [$ONLY]  both models  $(date -u)"
[ -f "$QGGUF" ] || { L "FATAL: $QGGUF missing"; exit 1; }
grep -q '"Q6_K", "Q6_K_L"' "$QGGUF" || { L "FATAL: $QGGUF not patched (Q6_K not in IMATRIX_EXCLUDE)"; exit 1; }

for row in "${ROWS[@]}"; do
  IFS='|' read -r CLEAN BF16DIR PUB <<<"$row"
  bf16="$GOOGLE/$BF16DIR"
  ggufdir="$bf16-GGUF"
  link="$GOOGLE/$CLEAN"
  repo_it="ManniX-ITA/gemma-4-A4B-98e-${PUB}-it"
  repo_gguf="ManniX-ITA/gemma-4-A4B-98e-${PUB}-it-GGUF"
  ollama="mannix/gemma4-98e-${PUB}"
  f16="$ggufdir/${CLEAN}-F16.gguf"

  L "===== $PUB  (clean=$CLEAN) ====="
  [ -d "$bf16" ]    || { L "FATAL: bf16 $bf16 missing"; exit 1; }
  [ -d "$ggufdir" ] || { L "FATAL: ggufdir $ggufdir missing"; exit 1; }
  [ -f "$f16" ]     || { L "FATAL: clean F16 $f16 missing (would force reconvert into wrong name)"; exit 1; }
  ln -sfn "$BF16DIR" "$link"     # clean-named --model so output stems match published
  L "symlink $link -> $BF16DIR ; reuse F16 $(du -h "$f16"|cut -f1)"

  "$PY" "$QGGUF" \
      --model "$link" \
      --output-dir "$ggufdir" \
      --only "$ONLY" \
      --repo "$repo_gguf" \
      --base-model-id "$repo_it" \
      --ollama-target "$ollama" \
      --ollama-template gemma4-a4b \
      --ollama-no-latest \
      --sanity-check \
      --ngl 0 \
      --hf-token "$HF_TOKEN" \
      || { L "FATAL: republish failed for $PUB"; exit 1; }
  L "===== $PUB DONE ====="
done
L "###### REPUBLISH_NOIMAT_FLIPPED_DONE $(date -u) ######"
