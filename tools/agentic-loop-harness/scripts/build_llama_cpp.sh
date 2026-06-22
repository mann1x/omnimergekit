#!/usr/bin/env bash
# Build a pinned CUDA llama-server for the agentic-loop-harness.
#
# Standalone + reusable: install.sh (--mode build) calls this, but you can run it
# by hand. Prints the resulting binary path on the last line (BIN=<path>).
#
#   scripts/build_llama_cpp.sh [--ref <git-ref>] [--arch <cuda-arch>] \
#       [--src-dir <dir>] [--jobs <N>] [--no-cuda]
#
# Defaults: ref=b9700 (the ref this project validated for Gemma-4 reasoning flags),
# arch autodetected from nvidia-smi, src-dir=./.llama.cpp, jobs=nproc.
set -euo pipefail

REF="b9700"
ARCH=""
SRC_DIR="$(pwd)/.llama.cpp"
JOBS="$(nproc 2>/dev/null || echo 4)"
CUDA=1

while [ $# -gt 0 ]; do
  case "$1" in
    --ref)     REF="$2"; shift 2 ;;
    --arch)    ARCH="$2"; shift 2 ;;
    --src-dir) SRC_DIR="$2"; shift 2 ;;
    --jobs)    JOBS="$2"; shift 2 ;;
    --no-cuda) CUDA=0; shift ;;
    *) echo "build_llama_cpp.sh: unknown arg $1" >&2; exit 2 ;;
  esac
done

command -v git   >/dev/null || { echo "ERROR: git not found" >&2; exit 1; }
command -v cmake >/dev/null || { echo "ERROR: cmake not found (apt install cmake)" >&2; exit 1; }

# Autodetect CUDA arch (e.g. 86=3090, 89=4090, 120=Blackwell PRO 6000) if not given.
if [ "$CUDA" = 1 ] && [ -z "$ARCH" ]; then
  if command -v nvidia-smi >/dev/null; then
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
    ARCH="${CC:-86}"
  else
    ARCH="86"
  fi
fi

echo ">>> llama.cpp build: ref=$REF cuda=$CUDA arch=${ARCH:-n/a} src=$SRC_DIR jobs=$JOBS" >&2

if [ ! -d "$SRC_DIR/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$SRC_DIR" >&2
fi
git -C "$SRC_DIR" fetch --tags --depth 1 origin "$REF" >&2 2>/dev/null || git -C "$SRC_DIR" fetch origin >&2
git -C "$SRC_DIR" checkout "$REF" >&2

CMAKE_FLAGS=(-B "$SRC_DIR/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF)
if [ "$CUDA" = 1 ]; then
  CMAKE_FLAGS+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH")
fi
cmake "${CMAKE_FLAGS[@]}" -S "$SRC_DIR" >&2
cmake --build "$SRC_DIR/build" --target llama-server -j "$JOBS" >&2

BIN="$SRC_DIR/build/bin/llama-server"
[ -x "$BIN" ] || { echo "ERROR: build produced no llama-server at $BIN" >&2; exit 1; }
echo ">>> built: $BIN" >&2
echo "BIN=$BIN"
