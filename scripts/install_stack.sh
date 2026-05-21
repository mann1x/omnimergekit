#!/bin/bash
# install_stack.sh — read EVAL_PROTOCOL v3 stack.lock.yaml and install all
# components. Idempotent — re-running is safe. Works on solidpc + cloud pods.
#
# Usage: bash install_stack.sh <path-to-stack.lock.yaml>
#
# Each component is installed into the conda env named in stack.lock.yaml.
# Patches are applied with sentinel-line checks so we know whether they're live.
#
# Exit: 0 success, 1 env not found, 2 install failed, 3 patch failed.

set -euo pipefail

STACK="${1:-}"
if [[ -z "$STACK" || ! -f "$STACK" ]]; then
    echo "usage: $0 <stack.lock.yaml>" >&2
    exit 1
fi

# Helpers
yq_get() { python3 -c "import yaml,sys; print(yaml.safe_load(open('$STACK'))$1)"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

NAME=$(yq_get "['name']")
VER=$(yq_get "['version']")
log "==== stack: ${NAME}@${VER} ===="

# Pick conda binary
CONDA=${CONDA_EXE:-/root/anaconda3/bin/conda}
[[ -x "$CONDA" ]] || CONDA=/opt/conda/bin/conda
[[ -x "$CONDA" ]] || { echo "conda not found"; exit 1; }
CONDA_ROOT=$(dirname $(dirname "$CONDA"))

run_in_env() {
    local env="$1"; shift
    local py="$CONDA_ROOT/envs/$env/bin/python"
    local pip="$CONDA_ROOT/envs/$env/bin/pip"
    if [[ ! -x "$py" ]]; then
        log "  env '$env' not found — creating with python 3.11"
        "$CONDA" create -y -n "$env" python=3.11 >/dev/null
    fi
    "$pip" "$@"
}

# ---- vLLM ----
VLLM_INSTALL=$(yq_get "['components']['vllm']['pip']")
VLLM_ENV=$(yq_get "['components']['vllm']['env']")
log "[vllm] env=${VLLM_ENV}  install: ${VLLM_INSTALL}"
run_in_env "$VLLM_ENV" install -q $VLLM_INSTALL

# Cherry-picks
N_CP=$(python3 -c "import yaml; d=yaml.safe_load(open('$STACK')); print(len(d['components']['vllm'].get('cherry_picks',[])))")
if [[ "$N_CP" != "0" ]]; then
    log "[vllm] $N_CP cherry-pick(s) declared — handled by separate cherry-pick script (TODO: integrate)"
fi

# ---- lm_eval ----
LME_INSTALL=$(yq_get "['components']['lm_eval']['pip']")
LME_ENV=$(yq_get "['components']['lm_eval']['env']")
log "[lm_eval] env=${LME_ENV}  install: ${LME_INSTALL}"
run_in_env "$LME_ENV" install -q "$LME_INSTALL"

# Patch: Fix-A reasoning_content fallback
PATCH_TARGET="$CONDA_ROOT/envs/$LME_ENV/lib/python3.11/site-packages/lm_eval/models/openai_completions.py"
SENTINEL=$(yq_get "['components']['lm_eval']['patches'][0]['sentinel_line']")
if grep -qF "$SENTINEL" "$PATCH_TARGET" 2>/dev/null; then
    log "[lm_eval] Fix-A already applied (sentinel found)"
else
    log "[lm_eval] applying Fix-A reasoning_content fallback"
    # Locate the parse_generations content extraction. The line we patch is the
    # `content = msg.get("content", "")` (or similar) in openai_completions.py
    # — append `or msg.get("reasoning_content")` to that fallback. Sentinel
    # check above will exit early on subsequent runs.
    python3 - "$PATCH_TARGET" <<'PY'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
# pattern: content = msg.get("content"...)  →  content = msg.get("content"...) or msg.get("reasoning_content")
new = re.sub(
    r'(content\s*=\s*msg\.get\(\s*["\']content["\']\s*(?:,\s*[^)]*)?\))(?!\s*or\s+msg\.get)',
    r'\1 or msg.get("reasoning_content")',
    src
)
if new == src:
    print("WARN: Fix-A pattern not matched — manual patch required at openai_completions.parse_generations")
    sys.exit(3)
p.write_text(new)
print("OK Fix-A applied")
PY
fi
log "[lm_eval] verify Fix-A sentinel:"
grep -c "$SENTINEL" "$PATCH_TARGET" || { echo "Fix-A sentinel missing"; exit 3; }

# ---- modelopt ----
MOPT_INSTALL=$(yq_get "['components']['modelopt']['pip']")
MOPT_ENV=$(yq_get "['components']['modelopt']['env']")
log "[modelopt] env=${MOPT_ENV}  install: ${MOPT_INSTALL}"
run_in_env "$MOPT_ENV" install -q "$MOPT_INSTALL"

# ---- omnimergekit ----
OMK_SHA=$(yq_get "['components']['omnimergekit_eval']['pinned_sha']")
if [[ "$OMK_SHA" != "TBD-on-promotion" ]]; then
    log "[omnimergekit] expected SHA: $OMK_SHA"
    CUR=$(cd /shared/dev/omnimergekit && git rev-parse HEAD 2>/dev/null || echo "")
    if [[ "$CUR" != "$OMK_SHA" ]]; then
        echo "WARN: /shared/dev/omnimergekit at $CUR, expected $OMK_SHA" >&2
    fi
fi

log "==== install complete — STACK ${NAME}@${VER} ===="
log "next: run omk_canary.py to verify before any cohort eval"
