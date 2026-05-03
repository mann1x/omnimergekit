#!/bin/bash
# Omnimerge v4: 3-source same-base frankenmerge over Qwen/Qwen3.6-27B.
# Sources: rico03 Claude-Opus reasoning distill (0.40), ValiantLabs Esper3.1 (0.35),
#          kai-os Opus reasoning LoRA merged into Qwen3.6 base (0.25).
# Method: Omnimerge_v2 (OBIM-lite + DAREx q=0.75 + EMR), density 0.53, seed 42.
#
# Pre-reqs (verify before running):
# - /workspace/base/qwen3.6-27b (Qwen/Qwen3.6-27B base)
# - /workspace/hf_models/rico03-claude-distill (rico03 fine-tune, ~55 GB)
# - /workspace/hf_models/valiant-esper3.1 (Esper3.1 fine-tune, ~55 GB)
# - /workspace/anchor_merged/Qwen3.6-27B-anchor (kai-os LoRA merged into Qwen3.6 base, ~52 GB)
# - /workspace/dare_ties_merge.py (merger)
#
set -euo pipefail

WORKSPACE=/workspace
BASE="$WORKSPACE/base/qwen3.6-27b"
S1="$WORKSPACE/hf_models/rico03-claude-distill"
S2="$WORKSPACE/hf_models/valiant-esper3.1"
S3="$WORKSPACE/anchor_merged/Qwen3.6-27B-anchor"
OUT="$WORKSPACE/merged/Qwen3.6-27B-Omnimerge-v4"
LOG="$WORKSPACE/logs/merge_v4.log"

mkdir -p "$WORKSPACE/merged" "$WORKSPACE/logs"

echo "=== v4 merge starting $(date -Iseconds) ===" | tee "$LOG"
echo "BASE: $BASE" | tee -a "$LOG"
echo "S1 (0.40): $S1" | tee -a "$LOG"
echo "S2 (0.35): $S2" | tee -a "$LOG"
echo "S3 (0.25): $S3" | tee -a "$LOG"
echo "OUT: $OUT" | tee -a "$LOG"
echo | tee -a "$LOG"

for d in "$BASE" "$S1" "$S2" "$S3"; do
    if [ ! -d "$d" ]; then echo "MISSING: $d" | tee -a "$LOG"; exit 1; fi
done

python3 /workspace/dare_ties_merge.py \
    --base "$BASE" \
    --source "$S1" \
    --source "$S2" \
    --source "$S3" \
    --weights 0.40,0.35,0.25 \
    --method omnimerge_v2 \
    --density 0.53 \
    --darex-q 0.75 \
    --seed 42 \
    --shard-size 5 \
    --output "$OUT" \
    2>&1 | tee -a "$LOG"

# Tokenizer override from base (merger default is to copy from S1)
echo "=== tokenizer override v4 ===" | tee -a "$LOG"
cp -fv "$BASE/tokenizer.json" "$OUT/tokenizer.json" | tee -a "$LOG"
cp -fv "$BASE/tokenizer_config.json" "$OUT/tokenizer_config.json" | tee -a "$LOG"
[ -f "$BASE/chat_template.jinja" ] && cp -fv "$BASE/chat_template.jinja" "$OUT/chat_template.jinja" | tee -a "$LOG"

echo "=== v4 merge done $(date -Iseconds) ===" | tee -a "$LOG"
du -sh "$OUT" | tee -a "$LOG"
ls -la "$OUT/" | tee -a "$LOG"
