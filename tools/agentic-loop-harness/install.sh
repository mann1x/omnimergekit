#!/usr/bin/env bash
# =============================================================================
# agentic-loop-harness installer  (autonomous, 3 setup modes)
# =============================================================================
# Installs the Python package + its single dependency, and arranges a
# llama-server one of three ways:
#
#   --mode build         build a pinned CUDA llama-server from source (default).
#   --mode byo-binary    use a llama-server binary you already have.
#   --mode byo-endpoint  use an already-running OpenAI-compatible endpoint
#                        (no binary; the harness runs in backend=endpoint mode).
#
# Examples:
#   ./install.sh --mode build                       # build pinned llama.cpp
#   ./install.sh --mode build --llama-ref b9700 --cuda-arch 120
#   ./install.sh --mode byo-binary --llama-server-bin /opt/llama.cpp/build/bin/llama-server
#   ./install.sh --mode byo-endpoint --endpoint http://127.0.0.1:8000
#
# Other flags:
#   --no-venv        install into the active Python instead of a local .venv
#   --python PATH    python interpreter to use (default: python3)
#
# Writes a `.env` with LLAMA_SERVER_BIN (modes build / byo-binary) so subsequent
# harness runs find the server automatically:  `source .env`.
#
# Secrets policy: this script takes NO tokens or keys and hardcodes none. It only
# clones the public llama.cpp and pip-installs from this directory.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

MODE="build"
LLAMA_REF="b9700"
CUDA_ARCH=""
LLAMA_BIN=""
ENDPOINT=""
USE_VENV=1
PYTHON="python3"

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)            MODE="$2"; shift 2 ;;
    --llama-ref)       LLAMA_REF="$2"; shift 2 ;;
    --cuda-arch)       CUDA_ARCH="$2"; shift 2 ;;
    --llama-server-bin) LLAMA_BIN="$2"; shift 2 ;;
    --endpoint)        ENDPOINT="$2"; shift 2 ;;
    --no-venv)         USE_VENV=0; shift ;;
    --python)          PYTHON="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "install.sh: unknown arg $1 (try --help)" >&2; exit 2 ;;
  esac
done

case "$MODE" in build|byo-binary|byo-endpoint) ;; *)
  echo "ERROR: --mode must be build | byo-binary | byo-endpoint" >&2; exit 2 ;;
esac

echo "==> agentic-loop-harness install  (mode=$MODE)"

# --- 1. Python package + dependency -----------------------------------------
PYBIN="$PYTHON"
if [ "$USE_VENV" = 1 ]; then
  echo "==> creating virtualenv at $HERE/.venv"
  "$PYTHON" -m venv .venv
  PYBIN="$HERE/.venv/bin/python"
fi
"$PYBIN" -m pip install -U pip >/dev/null
echo "==> pip install -e . (pulls PyYAML)"
"$PYBIN" -m pip install -e .
"$PYBIN" -c "import agentic_loop_harness, yaml; print('   package import OK, v'+agentic_loop_harness.__version__)"

# --- 2. server provisioning per mode ----------------------------------------
ENVFILE="$HERE/.env"
: > "$ENVFILE"

case "$MODE" in
  build)
    echo "==> building pinned llama-server (ref=$LLAMA_REF)"
    ARGS=(--ref "$LLAMA_REF" --src-dir "$HERE/.llama.cpp")
    [ -n "$CUDA_ARCH" ] && ARGS+=(--arch "$CUDA_ARCH")
    BIN_LINE="$(bash scripts/build_llama_cpp.sh "${ARGS[@]}" | tail -1)"
    BIN="${BIN_LINE#BIN=}"
    [ -x "$BIN" ] || { echo "ERROR: build did not yield a llama-server" >&2; exit 1; }
    echo "export LLAMA_SERVER_BIN=\"$BIN\"" >> "$ENVFILE"
    echo "==> built: $BIN"
    ;;
  byo-binary)
    [ -n "$LLAMA_BIN" ] || { echo "ERROR: --mode byo-binary needs --llama-server-bin PATH" >&2; exit 2; }
    [ -x "$LLAMA_BIN" ] || { echo "ERROR: not executable: $LLAMA_BIN" >&2; exit 1; }
    echo "export LLAMA_SERVER_BIN=\"$LLAMA_BIN\"" >> "$ENVFILE"
    echo "==> using your llama-server: $LLAMA_BIN"
    ;;
  byo-endpoint)
    [ -n "$ENDPOINT" ] || { echo "ERROR: --mode byo-endpoint needs --endpoint URL" >&2; exit 2; }
    echo "# backend=endpoint: set server.backend=endpoint + server.endpoint in your profile" >> "$ENVFILE"
    echo "export AGENTIC_LOOP_ENDPOINT=\"$ENDPOINT\"" >> "$ENVFILE"
    echo "==> will drive existing endpoint: $ENDPOINT"
    echo "    (set 'backend: endpoint' and 'endpoint: $ENDPOINT' in your run profile,"
    echo "     or pass --backend endpoint --endpoint $ENDPOINT at run time)"
    ;;
esac

# --- 3. summary -------------------------------------------------------------
cat <<EOF

==> install complete.
    python : $PYBIN
    env    : $ENVFILE  (source it to export LLAMA_SERVER_BIN / endpoint)

next:
    source .env
    # edit profiles/gemma4.example.yaml: point model.gguf at your GGUF
    $( [ "$USE_VENV" = 1 ] && echo ".venv/bin/agentic-loop-harness" || echo "agentic-loop-harness" ) \\
        --profile profiles/gemma4.example.yaml
EOF
