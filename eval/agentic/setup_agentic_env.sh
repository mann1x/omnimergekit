#!/usr/bin/env bash
# setup_agentic_env.sh — the agentic-harness env, single-sourced.
#
# Mirrors the ensure_*_env() contract in eval/pod_runners/setup_conda_envs.sh:
# NAMED env, requirements-pinned, idempotent, referenced by ABSOLUTE path.
# Kept separate from ensure_omk_env because this stack is lm-eval-free and its
# pins must not be dragged into the omk env (§1.4.5 applies to BOTH).
#
# Two backends, same contract — conda where it exists (pods), venv where it
# does not (linode-blackswan-2 has /root/anaconda3/envs/ but NO conda binary;
# the envs there are standalone interpreters used by absolute path).
set -uo pipefail
_SAE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_SAE_DIR/../.." && pwd)"
REQ="$REPO_ROOT/requirements-agentic.txt"
NAME="${1:-agentic}"
ENV_ROOT="${OMK_AGENTIC_ENV_ROOT:-/mnt/sdc/harness}"
PFX="${OMK_AGENTIC_ENV:-$ENV_ROOT/venv}"

[ -f "$REQ" ] || { echo "FAIL: missing $REQ"; exit 1; }

if [ ! -x "$PFX/bin/python" ]; then
    echo "[agentic] create env at $PFX"
    if command -v conda >/dev/null 2>&1; then
        conda create -n "$NAME" python=3.12 -y 2>&1 | tail -3
        PFX="$(conda info --base)/envs/$NAME"
    else
        echo "[agentic] no conda on this host — venv backend (python3.12)"
        mkdir -p "$(dirname "$PFX")"
        /usr/bin/python3.12 -m venv "$PFX" || { echo "FAIL: venv create (python3.12-venv installed?)"; exit 1; }
    fi
    "$PFX/bin/python" -m pip install --quiet --upgrade pip setuptools wheel
else
    echo "[agentic] env exists at $PFX"
fi

"$PFX/bin/python" -m pip install --quiet -r "$REQ" 2>&1 | tail -5
# inspect_ai's OpenAI-compatible provider is an OPTIONAL extra; without it the
# run dies at dispatch with "requires optional dependencies", after the model
# has already loaded. Pinned in requirements-agentic.txt, asserted here.
"$PFX/bin/python" - <<'PY'
import importlib.metadata as m
for p in ("inspect_ai", "inspect_evals", "openai"):
    print("  agentic env: %-14s %s" % (p, m.version(p)))
PY
echo "[agentic] OMK_AGENTIC_ENV=$PFX"
