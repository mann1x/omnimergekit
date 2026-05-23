#!/usr/bin/env bash
# apply_fix_a.sh — re-apply the refined Fix-A patch to lm-eval 0.4.11.
#
# Run on any fresh install of lm-eval to restore the refined Fix-A behavior
# documented in stack.lock.yaml.
#
# Usage:
#   apply_fix_a.sh                       # detects omnimergekit env automatically
#   apply_fix_a.sh /path/to/lm_eval      # explicit lm-eval pkg dir

set -euo pipefail
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LM_EVAL_DIR="${1:-}"
if [ -z "$LM_EVAL_DIR" ]; then
    for cand in \
        /root/anaconda3/envs/omnimergekit/lib/python3.11/site-packages/lm_eval \
        /opt/conda/lib/python3.11/site-packages/lm_eval \
        /workspace/miniconda/envs/omnimergekit/lib/python3.11/site-packages/lm_eval \
        /workspace/miniconda/lib/python3.11/site-packages/lm_eval ; do
        [ -d "$cand" ] && LM_EVAL_DIR="$cand" && break
    done
fi
[ -z "$LM_EVAL_DIR" ] && { echo "ERR: lm-eval dir not found" >&2; exit 2; }
echo "[fix-a] targeting $LM_EVAL_DIR"

TARGET="$LM_EVAL_DIR/models/openai_completions.py"
[ -f "$TARGET" ] || { echo "ERR: $TARGET missing" >&2; exit 2; }

# Idempotent check: refined Fix-A has the comment "(2026-05-14, refined 2026-05-21"
if grep -q "refined 2026-05-21" "$TARGET"; then
    echo "[fix-a] already applied"; exit 0
fi

# Snapshot
cp -p "$TARGET" "$TARGET.pre-fix-a"
# Apply with patch tool — fuzz allowed because lm-eval may have minor whitespace shifts
patch -p1 --no-backup-if-mismatch --fuzz=3 -d "$(dirname "$LM_EVAL_DIR")" \
    < "$PATCH_DIR/lm_eval_fix_a_reasoning_content_fallback.patch"

# Verify the sentinel
if grep -q "refined 2026-05-21" "$TARGET"; then
    echo "[fix-a] applied successfully"
else
    echo "[fix-a] ERROR: sentinel missing after patch — restoring backup" >&2
    mv "$TARGET.pre-fix-a" "$TARGET"
    exit 1
fi
# Bust .pyc so next import picks up the new bytecode
find "$LM_EVAL_DIR" -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
echo "[fix-a] done"
