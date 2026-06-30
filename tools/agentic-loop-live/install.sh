#!/usr/bin/env bash
# install.sh — set up agentic-loop-live. Core is stdlib-only; this just verifies the
# interpreter, optionally installs PyYAML (for YAML configs), and checks external
# prerequisites (opencode + a model server). You can also just run it from the checkout:
#   python -m agentic_loop_live --help
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "[install] python: $($PY --version 2>&1)"

if [ "${WITH_YAML:-1}" = "1" ]; then
    echo "[install] installing optional PyYAML (YAML configs); set WITH_YAML=0 to skip"
    "$PY" -m pip install --quiet "PyYAML>=5.1" || echo "[install] WARN: PyYAML install failed (JSON configs still work)"
fi

# optional editable install so `agentic-loop-live` is on PATH
if [ "${EDITABLE:-0}" = "1" ]; then
    "$PY" -m pip install -e .
fi

echo "[install] smoke check:"
"$PY" -m agentic_loop_live fixtures

echo
echo "[install] external prerequisites (install separately if missing):"
command -v opencode >/dev/null 2>&1 && echo "  - opencode: $(command -v opencode)" || echo "  - opencode: NOT FOUND -> https://opencode.ai"
command -v ollama   >/dev/null 2>&1 && echo "  - ollama:   $(command -v ollama)"   || echo "  - ollama:   (optional backend) not found"
echo "  - a llamafile binary OR llama.cpp llama-server (optional backends): point to it via backend.bin"
echo
echo "[install] done. Try:  $PY -m agentic_loop_live run --config config.example.yaml"
