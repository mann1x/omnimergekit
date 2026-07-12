#!/usr/bin/env bash
# upload_std16_hf.sh — publish STD16 (the chosen no-DERN release) to the live HF repos.
#   bf16 weights  -> ManniX-ITA/gemma-4-A4B-98e-v7-coder-it
#   13 loop-clean GGUF tiers + F16 + imatrix.dat -> ...-v7-coder-it-GGUF
# Loop/untested tiers (Q3_K_S, Q3_K_XL, Q2_K_L, qat-Q4_0, CD-qat-Q4_K_M, CD-IQ2_NL*) are
# DELIBERATELY NOT published — they fail the 48-seed loop gate. NVFP4A16 is built separately.
# hf CLI + existing `hf auth login` (or HF_TOKEN). hf_transfer for speed. Per-file, resumable.
set -uo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
BM=/srv/ml
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
BF=/mnt/sdc/ml/t223_fk/STD16-bf16
STEM=gemma-4-A4B-98e-v7-coder-it
REPO_BF=ManniX-ITA/gemma-4-A4B-98e-v7-coder-it
REPO_GG=ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF
WORK=/mnt/sdc/ml/std16_gate/publish
mkdir -p "$WORK"
LOG="$WORK/upload_hf.log"
exec >>"$LOG" 2>&1
ts(){ date -u +%T; }

# auth preflight — fail loud, upload nothing if not logged in
hf auth whoami >/dev/null 2>&1 || { echo "FATAL: hf not authenticated (run 'hf auth login' or export HF_TOKEN)"; exit 1; }
echo "==================== STD16 HF publish start $(ts) UTC  user=$(hf auth whoami 2>/dev/null) ===================="

CLEAN=(Q8_0 Q6_K_L Q6_K Q5_K_L Q5_K_M Q4_K_L Q4_K_M Q4_K_S IQ4_NL IQ4_XS Q3_K_L Q3_K_M CD-Q2_K)

up(){ # repo local_path path_in_repo
  local repo="$1" lp="$2" pir="$3" i
  [ -e "$lp" ] || { echo "[$(ts)] MISSING $lp (skip)"; return 0; }
  for i in 1 2 3 4 5; do
    echo "[$(ts)] upload try $i: $pir -> $repo"
    if hf upload "$repo" "$lp" "$pir" --repo-type model; then
      echo "[$(ts)] OK $pir"; return 0
    fi
    echo "[$(ts)] retry $pir in 30s"; sleep 30
  done
  echo "[$(ts)] !!! FAILED after 5 tries: $pir"
}

echo "===== bf16 weights -> $REPO_BF ====="
# upload each bf16 file (shards + configs + tokenizer); README handled separately
for f in "$BF"/*.safetensors "$BF"/*.json; do
  [ -e "$f" ] || continue
  up "$REPO_BF" "$f" "$(basename "$f")"
done

echo "===== 13 loop-clean GGUF tiers + F16 + imatrix -> $REPO_GG ====="
for T in "${CLEAN[@]}"; do
  up "$REPO_GG" "$GG/$STEM-$T.gguf" "$STEM-$T.gguf"
  [ -e "$GG/$STEM-$T.gguf.sha256" ] && up "$REPO_GG" "$GG/$STEM-$T.gguf.sha256" "$STEM-$T.gguf.sha256"
done
up "$REPO_GG" "$GG/$STEM-F16.gguf" "$STEM-F16.gguf"
up "$REPO_GG" "$GG/imatrix.dat" "imatrix.dat"

echo "[$(ts)] ==================== STD16 HF publish DONE $(ts) ===================="
echo "STD16_HF_PUBLISH_DONE"
