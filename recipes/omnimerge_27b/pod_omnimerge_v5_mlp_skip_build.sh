#!/bin/bash
# Omnimerge v5 (MLP-skip): proper from-scratch same-base merge for Qwen3.6.
# Uses the new --skip-patterns flag in dare_ties_merge.py (added 2026-04-29).
#
# Why MLP-skip:
#   Qwen3.6-27B's <think>-emission policy lives in MLP attractor structure.
#   Even 1-2% rel-L2 perturbations in mlp.{gate,up,down}_proj flip the policy
#   into "always open <think>, never close" — confirmed by v4 tensor-delta
#   forensics + v4-MLP-passthrough isolation (0% leak vs full-merge 80%).
#   See memory/feedback_qwen3_6_merge_policy_fragility.md for full evidence.
#
# Behavior of skip-patterns: tensors whose name contains any of the substrings
# get COPIED FROM BASE instead of merged. Everything else (attention,
# linear_attn, norms, embed/head) gets the full omnimerge_v2 treatment.
#
# Pre-reqs (verify before running):
# - /workspace/base/qwen3.6-27b
# - /workspace/hf_models/rico03-claude-distill
# - /workspace/hf_models/valiant-esper3.1
# - /workspace/anchor_merged/Qwen3.6-27B-anchor (kai-os LoRA merged into base)
# - /workspace/dare_ties_merge.py (with --skip-patterns support — push the
#   patched version from solidpc scripts/ before running)
#
set -euo pipefail

WORKSPACE=/workspace
BASE="$WORKSPACE/base/qwen3.6-27b"
S1="$WORKSPACE/hf_models/rico03-claude-distill"
S2="$WORKSPACE/hf_models/valiant-esper3.1"
S3="$WORKSPACE/anchor_merged/Qwen3.6-27B-anchor"
OUT="$WORKSPACE/merged/Qwen3.6-27B-Omnimerge-v5-mlp-skip"
LOG="$WORKSPACE/logs/merge_v5.log"

mkdir -p "$WORKSPACE/merged" "$WORKSPACE/logs"

echo "=== v5 mlp-skip merge starting $(date -Iseconds) ===" | tee "$LOG"
echo "BASE: $BASE" | tee -a "$LOG"
echo "S1 (0.40): $S1" | tee -a "$LOG"
echo "S2 (0.35): $S2" | tee -a "$LOG"
echo "S3 (0.25): $S3" | tee -a "$LOG"
echo "OUT: $OUT" | tee -a "$LOG"
echo "MLP-skip: mlp.gate_proj, mlp.up_proj, mlp.down_proj copied from base" | tee -a "$LOG"
echo | tee -a "$LOG"

for d in "$BASE" "$S1" "$S2" "$S3"; do
    if [ ! -d "$d" ]; then echo "MISSING: $d" | tee -a "$LOG"; exit 1; fi
done

if ! grep -q "skip-patterns" /workspace/dare_ties_merge.py; then
    echo "ERROR: dare_ties_merge.py on pod is the OLD version. Push the patched one from solidpc first." | tee -a "$LOG"
    exit 1
fi

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
    --skip-patterns 'mlp.gate_proj,mlp.up_proj,mlp.down_proj' \
    --output "$OUT" \
    2>&1 | tee -a "$LOG"

# Tokenizer override from base (merger default copies from S1)
echo "=== tokenizer override v5 ===" | tee -a "$LOG"
cp -fv "$BASE/tokenizer.json" "$OUT/tokenizer.json" | tee -a "$LOG"
cp -fv "$BASE/tokenizer_config.json" "$OUT/tokenizer_config.json" | tee -a "$LOG"
[ -f "$BASE/chat_template.jinja" ] && cp -fv "$BASE/chat_template.jinja" "$OUT/chat_template.jinja" | tee -a "$LOG"

echo "=== v5 merge done $(date -Iseconds) ===" | tee -a "$LOG"
du -sh "$OUT" | tee -a "$LOG"
ls -la "$OUT/" | tee -a "$LOG"
