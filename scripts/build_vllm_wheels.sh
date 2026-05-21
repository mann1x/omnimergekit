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
WHEELS_DIR=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/wheels/gemma4-moe-stack-v2
BRANCH=gemma4-moe-stack-v2
PY=${PY:-/root/anaconda3/envs/vllm/bin/python}

DEFAULT_ARCHES=(sm86 sm89 sm90 sm100 sm120)
MULTI=0
SELECTED=()

for arg in "$@"; do
    case "$arg" in
        --multi) MULTI=1 ;;
        sm[0-9]*) SELECTED+=("$arg") ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done
[[ ${#SELECTED[@]} -eq 0 ]] && SELECTED=("${DEFAULT_ARCHES[@]}")

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

# Sanity check: Fix-E patch is present
if ! grep -q "Fix E (parser hardening" vllm/reasoning/gemma4_reasoning_parser.py; then
    echo "ERROR: Fix-E patch not detected in $BRANCH — abort" >&2
    exit 2
fi
log "Fix-E parser patch verified present"

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
    "$PY" setup.py bdist_wheel 2>&1 | tail -50
    local t1=$(date +%s)
    log "$arch built in $((t1-t0))s"

    # Tag wheel with arch suffix and stash
    local W=$(ls -t dist/*.whl | head -1)
    [[ -z "$W" ]] && { echo "no wheel produced for $arch"; return 1; }
    local BASE=$(basename "$W" .whl)
    local OUT="${WHEELS_DIR}/${BASE}+${arch}.whl"
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
    "$PY" setup.py bdist_wheel 2>&1 | tail -50
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
    echo "# gemma4-moe-stack@2 wheel manifest"
    echo "# generated: $(date -Iseconds)"
    echo "# vllm-source HEAD: $SHA  (branch $BRANCH)"
    echo "# fix-E parser cherry-pick: 3d92852eb (originally a39e23ed0)"
    echo "# base: vllm main HEAD with #42250 closure fix, #43223 NvFP4 MoE,"
    echo "#       #38939 routed-experts API, #42664 reasoning normalize"
    echo ""
    echo "# arch  size      wheel"
    ls -la "$WHEELS_DIR"/*.whl 2>/dev/null | awk '{print $5"\t"$NF}'
} > "$MANIFEST"

log "==== wheels ready in $WHEELS_DIR ===="
ls -la "$WHEELS_DIR"
