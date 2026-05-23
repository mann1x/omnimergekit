#!/bin/bash
# build_vllm_wheels.sh — produce per-arch vLLM wheels for the
# gemma4-moe-stack@2 stack lock.
#
# Stack contents:
#   - vLLM main HEAD (which already includes #42250 closure fix,
#     #43223 NvFP4 MoE routing, #38939 routed-experts API,
#     #42664 reasoning_content normalize)
#   - PLUS cherry-pick: Fix-E parser hardening (3d92852eb)
#     — surface reasoning as content on half-open thinking
#
# Builds one wheel per CUDA compute capability so pods only download
# what their GPU actually uses:
#   sm86  — RTX 3090, A40
#   sm89  — RTX 4090, L40, RTX 6000 Ada, A6000-Ada
#   sm90  — H100, H200
#   sm100 — B100, B200 (datacenter Blackwell)
#   sm120 — RTX 50-series (consumer Blackwell)
#
# Usage:
#   bash build_vllm_wheels.sh                         # all arches
#   bash build_vllm_wheels.sh sm86                    # just 3090
#   bash build_vllm_wheels.sh sm86 sm89 sm90          # 3 arches
#   bash build_vllm_wheels.sh --multi                 # one fat wheel (all)
#
# Expected wall: ~25-45 min per arch on solidpc (16-core, MAX_JOBS=12).
# Output: $WHEELS_DIR/vllm-<ver>+gemma4moe-cp311-cp311-linux_x86_64.smXX.whl
#
# Prereqs: CUDA toolkit 12.8, gcc 11+, python 3.11, ~30GB RAM.
# Do NOT run while GPU evals are in flight — the build is CPU-bound
# but pulls ~12 cores and may compete with eval orchestrator.

set -euo pipefail

VLLM_SRC=/srv/dev-disk-by-label-opt/dev/vllm-source
WHEELS_ROOT=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/wheels
PY=${PY:-/root/anaconda3/envs/vllm/bin/python}

# Defaults — preserved for backwards compatibility.
BRANCH="${BRANCH_OVERRIDE:-gemma4-moe-stack-v2}"
SKIP_FIXE_CHECK=0

DEFAULT_ARCHES=(sm86 sm89 sm90 sm100 sm120)
MULTI=0
SELECTED=()

# Parse args. Accepts:
#   sm86 sm89 ...         (positional arch tokens)
#   --multi               (one fat wheel)
#   --branch <ref>        (override branch/tag/sha; e.g. v0.20.2)
#   --skip-fix-e-check    (don't require Fix-E patch in source — for bisection)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --multi) MULTI=1; shift ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --branch=*) BRANCH="${1#*=}"; shift ;;
        --skip-fix-e-check) SKIP_FIXE_CHECK=1; shift ;;
        sm[0-9]*) SELECTED+=("$1"); shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ ${#SELECTED[@]} -eq 0 ]] && SELECTED=("${DEFAULT_ARCHES[@]}")

# Per-branch wheels dir — keep stack@2 wheels isolated from experimental builds.
# Sanitize ref for use as a directory name (e.g. v0.20.2 → v0.20.2).
BRANCH_SLUG=$(echo "$BRANCH" | tr '/' '_')
WHEELS_DIR="$WHEELS_ROOT/$BRANCH_SLUG"

arch_to_dot() {
    local s="${1#sm}"
    echo "${s:0:1}.${s:1}"
}

mkdir -p "$WHEELS_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

cd "$VLLM_SRC"
git checkout "$BRANCH" 2>&1 | tail -3
SHA=$(git rev-parse --short HEAD)
log "vllm-source branch=$BRANCH  HEAD=$SHA"

# Pin compiler to conda-forge gcc-11 in the vllm env. Debian 11 system gcc
# is 10.2, but vllm's CMake requires >=11.0; without this the build picks
# up /usr/bin/gcc and aborts at the version check. The conda binaries are
# installed by `conda install -n vllm -c conda-forge gcc_linux-64=11.4
# gxx_linux-64=11 cmake ninja` (see EVAL_PROTOCOL §v3.3 step 6 prereqs).
CONDA_GCC=/root/anaconda3/envs/vllm/bin/x86_64-conda-linux-gnu-gcc
CONDA_GXX=/root/anaconda3/envs/vllm/bin/x86_64-conda-linux-gnu-g++
if [[ ! -x "$CONDA_GCC" || ! -x "$CONDA_GXX" ]]; then
    echo "ERROR: conda gcc-11 toolchain missing in vllm env" >&2
    echo "  install: conda install -n vllm -c conda-forge gcc_linux-64=11.4 gxx_linux-64=11 cmake ninja" >&2
    exit 3
fi
export CC="$CONDA_GCC"
export CXX="$CONDA_GXX"
# nvcc has its OWN host compiler discovery — without this it picks up
# Debian 11's /usr/bin/gcc → gcc-10, which ICEs on CUDA-13 + cccl's heavy
# C++20 templates (see vllm csrc/moe/permute_unpermute_kernels). CUDAHOSTCXX
# is what CMake reads to populate CMAKE_CUDA_HOST_COMPILER, which becomes
# nvcc's -ccbin flag. CUDA 13.2 supports gcc 11–14, conda gcc-11.4 works.
export CUDAHOSTCXX="$CONDA_GXX"
log "CC=$CC ($($CC -dumpversion))"
log "CXX=$CXX ($($CXX -dumpversion))"
log "CUDAHOSTCXX=$CUDAHOSTCXX (nvcc -ccbin will use this for host compiles)"

# Put the env's bin first so CMake's discovery sees conda's cmake/ninja too.
export PATH="/root/anaconda3/envs/vllm/bin:$PATH"

# ccache — cache compiled host C/C++ AND nvcc CUDA objects across arches and
# across reruns. Single persistent dir on backup_models so the cache survives
# pod-to-pod transfers via rsync if we ever want to seed an A100 build.
# Expected hit rate after the first full sm86 build: 30-60 % on subsequent
# arches (most of vllm's C++ doesn't depend on the CUDA arch). Without this,
# every arch re-compiles ~3500 host C++ TUs from scratch.
if ! command -v ccache >/dev/null 2>&1; then
    echo "ERROR: ccache not found on PATH" >&2
    exit 4
fi
# ccache stays shared across branches — most of vllm's C++ is branch-invariant,
# so the v0.20.2 build can recycle objects compiled for stack@2. Pinning the
# path to gemma4-moe-stack-v2 is intentional (it's where the original sm86 build
# already populated the cache; a fresh empty cache would cost 30–60 min more).
export CCACHE_DIR=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/wheels/gemma4-moe-stack-v2/ccache
mkdir -p "$CCACHE_DIR"
# Hash by content not mtime — conda envs touch binaries during install
export CCACHE_COMPILERCHECK=content
# nvcc/CMake generate timestamped wrappers; relax sloppiness to land hits
export CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime,include_file_ctime,locale
ccache --max-size=25G >/dev/null
ccache --zero-stats >/dev/null
log "ccache $(ccache --version | head -1 | awk '{print $3}') dir=$CCACHE_DIR (max 25G)"

# CMake launcher wiring — applied to every Extension/CUDA target via env
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
# Some vllm CMake paths still respect plain VLLM_CCACHE/USE_CCACHE
export USE_CCACHE=1

# Skip FetchContent network ops by pointing every external project at the
# already-downloaded .deps/*-src tree. First-time build downloaded these on
# 2026-05-21; subsequent rebuilds reuse them. GitHub's pack delivery for
# the NVIDIA/cutlass submodule under FlashMLA is rate-limited to ~1 MB/min
# from solidpc → without these env vars an sm86 configure takes 100+ min.
#
# Arch impact for sm86 (3090): DeepGEMM and FlashMLA kernels only build
# for sm90a (Hopper); qutlass only for sm100 (datacenter Blackwell). So
# for sm86 these are source-only — we still need the dir to exist so the
# CMake INCLUDE doesn't fail, but no kernels compile from them.
DEPS=/srv/dev-disk-by-label-opt/dev/vllm-source/.deps
if [[ -d "$DEPS/flashmla-src" ]]; then
    export FLASH_MLA_SRC_DIR="$DEPS/flashmla-src"
    export DEEPGEMM_SRC_DIR="$DEPS/deepgemm-src"
    export QUTLASS_SRC_DIR="$DEPS/qutlass-src"
    # NOTE: triton_kernels.cmake's env-var branch expects the DEEP python
    # subdir (per the in-file comment "directly set to the triton_kernels
    # python directory"). The git-fetch branch uses SOURCE_SUBDIR to scope.
    # Pointing at the repo root triggers `find_package(MLIR)` against the
    # full Triton CMakeLists.txt, which fails on systems without LLVM/MLIR.
    export TRITON_KERNELS_SRC_DIR="$DEPS/triton_kernels-src/python/triton_kernels/triton_kernels"
    export VLLM_FLASH_ATTN_SRC_DIR="$DEPS/vllm-flash-attn-src"
    log "FetchContent local-source: FLASH_MLA / DEEPGEMM / QUTLASS / TRITON_KERNELS / VLLM_FLASH_ATTN → $DEPS/*-src"
fi

# Sanity check: Fix-E patch is present (skip with --skip-fix-e-check for
# branches that predate Fix-E, e.g. v0.20.2 used as a bisection baseline).
if [[ "$SKIP_FIXE_CHECK" -eq 0 ]]; then
    if ! grep -q "Fix E (parser hardening" vllm/reasoning/gemma4_reasoning_parser.py 2>/dev/null; then
        echo "ERROR: Fix-E patch not detected in $BRANCH — abort" >&2
        echo "       (pass --skip-fix-e-check to build a pre-Fix-E baseline)" >&2
        exit 2
    fi
    log "Fix-E parser patch verified present"
else
    log "--skip-fix-e-check active; Fix-E presence NOT verified for $BRANCH"
fi

build_one_arch() {
    local arch=$1
    local dot=$(arch_to_dot "$arch")
    log "=== building $arch (CUDA arch $dot) ==="

    # Clean prior build artifacts (kernels are arch-specific)
    rm -rf build/ dist/ *.egg-info

    # Build flags
    export TORCH_CUDA_ARCH_LIST="$dot"
    export MAX_JOBS="${MAX_JOBS:-12}"
    export VLLM_TARGET_DEVICE=cuda
    export CMAKE_BUILD_TYPE=Release

    local t0=$(date +%s)
    # NOTE: do NOT pipe through `tail` — buffer trap hides real-time errors.
    # Full output is captured in the wrapper's log file already.
    "$PY" setup.py bdist_wheel 2>&1
    local t1=$(date +%s)
    log "$arch built in $((t1-t0))s"

    # Tag wheel with arch suffix in the version local-tag (PEP 440 compatible).
    # vLLM ships wheels named vllm-<ver>+g<sha>.cu132-cp311-cp311-linux_x86_64.whl;
    # inserting .<arch> AFTER the .cu132 local tag keeps the platform tag clean
    # (linux_x86_64) so pip will install — putting it after platform tag breaks
    # PEP 425 and pip rejects with "not a supported wheel on this platform".
    local W=$(ls -t dist/*.whl | head -1)
    [[ -z "$W" ]] && { echo "no wheel produced for $arch"; return 1; }
    local BASENAME=$(basename "$W")
    # Replace "-cp311-cp311-" boundary: insert .${arch} just before it.
    # If the existing version already ends with .cuNNN (local tag), append .<arch> to it.
    local OUTNAME=$(echo "$BASENAME" | sed -E "s/(-cp311-cp311-linux_x86_64\.whl)$/.${arch}\1/")
    local OUT="${WHEELS_DIR}/${OUTNAME}"
    mv "$W" "$OUT"
    log "  → $OUT  ($(du -h "$OUT" | cut -f1))"
}

build_multi() {
    local archlist=""
    for a in "${SELECTED[@]}"; do
        local d=$(arch_to_dot "$a")
        archlist="${archlist:+$archlist;}$d"
    done
    log "=== building multi-arch wheel (CUDA archs: $archlist) ==="

    rm -rf build/ dist/ *.egg-info

    export TORCH_CUDA_ARCH_LIST="$archlist"
    export MAX_JOBS="${MAX_JOBS:-12}"
    export VLLM_TARGET_DEVICE=cuda
    export CMAKE_BUILD_TYPE=Release

    local t0=$(date +%s)
    # NOTE: do NOT pipe through `tail` — buffer trap hides real-time errors.
    # Full output is captured in the wrapper's log file already.
    "$PY" setup.py bdist_wheel 2>&1
    local t1=$(date +%s)
    log "multi-arch built in $((t1-t0))s"

    local W=$(ls -t dist/*.whl | head -1)
    local BASE=$(basename "$W" .whl)
    local TAG=$(echo "${SELECTED[@]}" | tr ' ' '_')
    local OUT="${WHEELS_DIR}/${BASE}+multi_${TAG}.whl"
    mv "$W" "$OUT"
    log "  → $OUT  ($(du -h "$OUT" | cut -f1))"
}

if [[ $MULTI -eq 1 ]]; then
    build_multi
else
    for a in "${SELECTED[@]}"; do
        build_one_arch "$a"
    done
fi

# Write manifest
MANIFEST="$WHEELS_DIR/MANIFEST.txt"
{
    echo "# vllm wheel manifest"
    echo "# generated: $(date -Iseconds)"
    echo "# vllm-source HEAD: $SHA  (branch $BRANCH)"
    if [[ "$SKIP_FIXE_CHECK" -eq 0 ]]; then
        echo "# fix-E parser cherry-pick: present"
    else
        echo "# fix-E parser cherry-pick: SKIPPED (--skip-fix-e-check)"
    fi
    echo ""
    echo "# arch  size      wheel"
    ls -la "$WHEELS_DIR"/*.whl 2>/dev/null | awk '{print $5"\t"$NF}'
} > "$MANIFEST"

log "==== wheels ready in $WHEELS_DIR ===="
ls -la "$WHEELS_DIR"
