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

# Snapshot (restored if the apply fails to land the sentinel).
cp -p "$TARGET" "$TARGET.pre-fix-a"
# Apply via the robust string-replace patcher — NOT a context-diff. A `.patch`
# assumes a known starting state; fresh lm-eval installs ship the STOCK 0.4.11
# parse_generations form, which the unified diff can't match ("Hunk FAILED" —
# the 2026-05-27 day-burn). fix_a_lm_eval_patch.py anchors on the stock line,
# emits the refined Fix-A, and is idempotent. Pure-stdlib → any python works.
PYBIN="${OMK_PYTHON:-$(command -v python3 || command -v python)}"
"$PYBIN" "$PATCH_DIR/fix_a_lm_eval_patch.py" "$TARGET"

# Verify the sentinel landed; restore the snapshot on any failure.
if grep -q "refined 2026-05-21" "$TARGET"; then
    echo "[fix-a] applied successfully"
    rm -f "$TARGET.pre-fix-a"
else
    echo "[fix-a] ERROR: sentinel missing after apply — restoring backup" >&2
    mv "$TARGET.pre-fix-a" "$TARGET"
    exit 1
fi
# Bust .pyc so next import picks up the new bytecode
find "$LM_EVAL_DIR" -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
echo "[fix-a] done"
