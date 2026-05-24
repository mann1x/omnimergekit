#!/usr/bin/env bash
# pod_eval_bootstrap.sh — THE canonical on-pod bootstrap for omnimergekit
# eval / surgery work on a rented GPU pod.
#
# Run it ON the pod after landing the repo. Canonical one-liner:
#
#   git clone https://github.com/mann1x/omnimergekit /workspace/omnimergekit \
#     && HF_TOKEN=... bash /workspace/omnimergekit/eval/pod_eval_bootstrap.sh <flags>
#
# It is idempotent (re-run = no-op) and the single source of truth for the
# on-pod SYSTEM layer: apt deps, miniconda, repo, llama.cpp build, conda env(s),
# the solidpc-path symlink farm, HF pulls, and the lm-eval patches. The heavy
# version-pinned conda envs (omnimergekit-from-requirements, modelopt, vllm) are
# delegated to eval/pod_runners/setup_conda_envs.sh so the pins live in one place.
#
# The five former one-off bootstraps (c6v3lcb / reeval / gpt_oss / hep_sweep /
# router_recovery) are now thin wrappers in eval/pod_runners/ that just call this
# script with the right flags. See EVAL_PROTOCOL.md §2.5.
#
# Per SECURITY: never hardcode HF_TOKEN. Export it; this script fails loud if a
# pull is requested without it. NEVER run a secret scan in a git-commit bash block.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_LIB="$SCRIPT_DIR/pod_runners/setup_conda_envs.sh"

# ── defaults ─────────────────────────────────────────────────────────────────
APT_EXTRA=""
INSTALL_MINICONDA=0
DO_REPO=1
REPO_DIR=/workspace/omnimergekit
REPO_URL=https://github.com/mann1x/omnimergekit
BUILD_LLAMA=1
CUDA_ARCH=86
LLAMA_CURL=OFF
ENV_NAME=omnimergekit
DEPS=eval-full                 # eval-full | eval-augment | train | none
LMEVAL_EXTRAS="api,math,ifeval"
LMEVAL_VERSION="0.4.11"
PATCHES="unbound"              # csv: unbound,fix-a
WANT_VLLM=0
VLLM_VERSION="0.20.2"
WANT_MODELOPT=0
SYMLINK_FARM=0
TOKENIZER_LINK=""              # SRC:DST
HF_PULL=""                     # "repo:dir repo:dir ..."
HF_PULL_PATTERNS="*.safetensors *.json tokenizer* *.model *.jinja"
DRY_RUN=0

CONDA_ROOT=/workspace/miniconda
CONDA_ENVS="$CONDA_ROOT/envs"
LOG=/workspace/logs/pod_eval_bootstrap.log

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Flags:
  --apt-extra "PKGS"      extra apt packages (e.g. "libcurl4-openssl-dev zstd")
  --install-miniconda     install miniconda at /workspace/miniconda if absent
  --no-repo               skip omnimergekit clone/pull (controller rsynced it)
  --repo-dir DIR          omnimergekit dir (default /workspace/omnimergekit)
  --repo-url URL          omnimergekit git url
  --llama / --no-llama    build llama.cpp (default: build)
  --cuda-arch ARCH        CUDA arch for llama.cpp (default 86=3090; 89=4090; "86;89")
  --llama-curl            build llama.cpp with LLAMA_CURL=ON (default OFF)
  --env NAME              conda env to use/create (default omnimergekit)
  --deps PROFILE          eval-full | eval-augment | train | none (default eval-full)
                            eval-full   : requirements.txt + requirements-eval.txt (bare pod)
                            eval-augment: lean lm-eval stack into an existing torch env (rr)
                            train       : torch+transformers+bnb+datasets (Router-KD/surgery)
                            none        : ensure env exists, install nothing
  --lm-eval-extras CSV    lm-eval extras for eval-augment (default api,math,ifeval)
  --lm-eval-version VER   lm-eval pin for eval-augment (default 0.4.11)
  --patches CSV           lm-eval patches: unbound,fix-a (default unbound; eval-* only)
  --vllm-env              also create the vllm conda env (delegates to env lib)
  --vllm-version VER      vllm pin (default 0.20.2)
  --modelopt-env          also create the modelopt conda env (delegates to env lib)
  --symlink-farm          build the solidpc-path symlink farm (for eval_suite_*.sh)
  --tokenizer-link S:D    symlink tokenizer dir S -> D (with --symlink-farm)
  --hf-pull "R:D ..."     HF downloads, space-separated repo:localdir pairs (needs HF_TOKEN)
  --hf-pull-patterns "P"  include patterns for --hf-pull (default weights+config+tokenizer)
  --dry-run               print resolved config + planned steps, touch nothing
  -h, --help              this help
EOF
}

# ── arg parse ────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --apt-extra)        APT_EXTRA="$2"; shift 2 ;;
        --install-miniconda) INSTALL_MINICONDA=1; shift ;;
        --no-repo)          DO_REPO=0; shift ;;
        --repo-dir)         REPO_DIR="$2"; shift 2 ;;
        --repo-url)         REPO_URL="$2"; shift 2 ;;
        --llama)            BUILD_LLAMA=1; shift ;;
        --no-llama)         BUILD_LLAMA=0; shift ;;
        --cuda-arch)        CUDA_ARCH="$2"; shift 2 ;;
        --llama-curl)       LLAMA_CURL=ON; shift ;;
        --env)              ENV_NAME="$2"; shift 2 ;;
        --deps)             DEPS="$2"; shift 2 ;;
        --lm-eval-extras)   LMEVAL_EXTRAS="$2"; shift 2 ;;
        --lm-eval-version)  LMEVAL_VERSION="$2"; shift 2 ;;
        --patches)          PATCHES="$2"; shift 2 ;;
        --vllm-env)         WANT_VLLM=1; shift ;;
        --vllm-version)     VLLM_VERSION="$2"; shift 2 ;;
        --modelopt-env)     WANT_MODELOPT=1; shift ;;
        --symlink-farm)     SYMLINK_FARM=1; shift ;;
        --tokenizer-link)   TOKENIZER_LINK="$2"; shift 2 ;;
        --hf-pull)          HF_PULL="$2"; shift 2 ;;
        --hf-pull-patterns) HF_PULL_PATTERNS="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$DEPS" in eval-full|eval-augment|train|none) ;; *) echo "bad --deps: $DEPS" >&2; exit 2 ;; esac
ENV_PFX="$CONDA_ENVS/$ENV_NAME"
ENV_PY="$ENV_PFX/bin/python"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

if [ "$DRY_RUN" = 1 ]; then
    cat <<EOF
=== pod_eval_bootstrap.sh --dry-run ===
apt-extra        : ${APT_EXTRA:-(none)}
install-miniconda: $INSTALL_MINICONDA
repo             : $([ "$DO_REPO" = 1 ] && echo "$REPO_URL -> $REPO_DIR" || echo "(skip, --no-repo)")
llama.cpp        : $([ "$BUILD_LLAMA" = 1 ] && echo "build arch=$CUDA_ARCH curl=$LLAMA_CURL" || echo "(skip)")
conda env        : $ENV_NAME ($ENV_PFX)
deps profile     : $DEPS$([ "$DEPS" = eval-augment ] && echo " (lm-eval[$LMEVAL_EXTRAS]==$LMEVAL_VERSION)")
lm-eval patches  : $([ "$DEPS" = train ] && echo "(n/a for train)" || echo "$PATCHES")
vllm env         : $([ "$WANT_VLLM" = 1 ] && echo "yes (==$VLLM_VERSION)" || echo no)
modelopt env     : $([ "$WANT_MODELOPT" = 1 ] && echo yes || echo no)
symlink farm     : $([ "$SYMLINK_FARM" = 1 ] && echo "yes${TOKENIZER_LINK:+ tokenizer=$TOKENIZER_LINK}" || echo no)
hf-pull          : ${HF_PULL:-(none)}
hf-pull-patterns : $HF_PULL_PATTERNS
env lib          : $ENV_LIB $([ -f "$ENV_LIB" ] && echo "(found)" || echo "(MISSING!)")
HF_TOKEN         : $([ -n "${HF_TOKEN:-}" ] && echo set || echo "(unset)")
EOF
    [ -n "$HF_PULL" ] && [ -z "${HF_TOKEN:-}" ] && echo "WARN: --hf-pull set but HF_TOKEN unset — would fail at pull"
    echo "=== (dry run, nothing changed) ==="
    exit 0
fi

mkdir -p /workspace/logs /workspace/scripts
log "=== pod_eval_bootstrap START (env=$ENV_NAME deps=$DEPS llama=$BUILD_LLAMA) ==="

# ── 1. apt deps ──────────────────────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
log "1. apt deps"
apt-get update -qq
# sqlite3 is mandatory: EVAL_PROTOCOL §3.1 launch-check decodes the lm-eval cache.
apt-get install -y -qq cmake ninja-build build-essential git rsync wget curl ca-certificates sqlite3 ${APT_EXTRA} >/dev/null
log "   cmake $(cmake --version | head -1) | sqlite3 $(sqlite3 --version | awk '{print $1}')"

# ── 2. miniconda ─────────────────────────────────────────────────────────────
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
    if [ "$INSTALL_MINICONDA" = 1 ]; then
        log "2. install miniconda -> $CONDA_ROOT"
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
        bash /tmp/mc.sh -b -p "$CONDA_ROOT" && rm -f /tmp/mc.sh
    else
        log "FATAL: $CONDA_ROOT/bin/conda missing and --install-miniconda not given"; exit 1
    fi
else
    log "2. miniconda present"
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
export PYTHONDONTWRITEBYTECODE=1
export HF_XET_HIGH_PERFORMANCE=1
# /root/anaconda3/envs symlink: eval_suite_*.sh hardcode that path.
mkdir -p /root/anaconda3
[ -e /root/anaconda3/envs ] || ln -s "$CONDA_ENVS" /root/anaconda3/envs

# Source the shared env recipes (function defs only). Needed for apply_lmeval_patches
# and for ensure_omk_env / ensure_vllm_env / ensure_modelopt_env.
if [ -f "$ENV_LIB" ]; then
    # shellcheck disable=SC1090
    source "$ENV_LIB"
else
    log "WARN: env lib $ENV_LIB not found — eval-full / vllm / modelopt / patches unavailable"
fi

# ── 3. omnimergekit repo ─────────────────────────────────────────────────────
if [ "$DO_REPO" = 1 ]; then
    if [ ! -d "$REPO_DIR/.git" ]; then
        log "3. clone omnimergekit -> $REPO_DIR"
        git clone "$REPO_URL" "$REPO_DIR" 2>&1 | tail -2
    else
        log "3. omnimergekit present — git pull"
        git -C "$REPO_DIR" pull --ff-only 2>&1 | tail -2 || true
    fi
else
    log "3. repo step skipped (--no-repo)"
fi

# ── 4. llama.cpp build (CUDA) ────────────────────────────────────────────────
if [ "$BUILD_LLAMA" = 1 ]; then
    if [ ! -x /workspace/llama.cpp/build/bin/llama-server ]; then
        log "4. build llama.cpp (arch=$CUDA_ARCH curl=$LLAMA_CURL)"
        [ -d /workspace/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp 2>&1 | tail -2
        cd /workspace/llama.cpp
        cmake -B build -G Ninja \
            -DGGML_CUDA=ON \
            -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
            -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
            -DLLAMA_CURL="$LLAMA_CURL" \
            -DCMAKE_BUILD_TYPE=Release >/dev/null
        cmake --build build --target llama-server llama-quantize llama-imatrix --parallel 2>&1 | tail -3
        cd - >/dev/null
    else
        log "4. llama.cpp already built"
    fi
    ln -sfn /workspace/llama.cpp /opt/llama.cpp
    log "   llama-server: $(/opt/llama.cpp/build/bin/llama-server --version 2>&1 | head -1 || echo '?')"
else
    log "4. llama.cpp skipped (--no-llama)"
fi

# ── 5. conda env + deps ──────────────────────────────────────────────────────
log "5. conda env '$ENV_NAME' (deps=$DEPS)"
case "$DEPS" in
    eval-full)
        ensure_omk_env "$ENV_NAME"
        # ensure_omk_env applies 'unbound'; add fix-a only if explicitly requested.
        case "$PATCHES" in *fix-a*) apply_lmeval_patches "$ENV_PY" "fix-a" ;; esac
        ;;
    eval-augment)
        if [ ! -x "$ENV_PY" ]; then
            log "   create $ENV_NAME (py3.11) for augment"
            conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>&1 | tail -1 || true
            conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>&1 | tail -1 || true
            conda create -n "$ENV_NAME" python=3.11 -y 2>&1 | tail -3
        fi
        "$ENV_PFX/bin/pip" install --quiet --upgrade pip
        "$ENV_PFX/bin/pip" install --quiet \
            "lm-eval[${LMEVAL_EXTRAS}]==${LMEVAL_VERSION}" \
            'antlr4-python3-runtime==4.11.0' 'sympy>=1.13' 'math-verify>=0.9.0' \
            'langdetect>=1.0.9' 'immutabledict>=4.2.0' 'nltk>=3.9.1' \
            'tenacity==9.1.4' 'human-eval>=1.0.3' 'gguf==0.18.0' 'pyyaml>=6.0' 2>&1 | tail -3
        apply_lmeval_patches "$ENV_PY" "$PATCHES"
        ;;
    train)
        if [ ! -x "$ENV_PY" ]; then
            log "   create $ENV_NAME (py3.11) for train"
            conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>&1 | tail -1 || true
            conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>&1 | tail -1 || true
            conda create -n "$ENV_NAME" python=3.11 -y 2>&1 | tail -3
        fi
        "$ENV_PFX/bin/pip" install --quiet --upgrade pip
        "$ENV_PFX/bin/pip" install --quiet torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
        "$ENV_PFX/bin/pip" install --quiet \
            'transformers==5.5.0' datasets bitsandbytes accelerate safetensors hf_transfer huggingface_hub gguf 2>&1 | tail -2
        ;;
    none)
        [ -x "$ENV_PY" ] || { log "   create $ENV_NAME (py3.11)"; conda create -n "$ENV_NAME" python=3.11 -y 2>&1 | tail -3; }
        ;;
esac

# ── 6. optional heavy envs (delegated) ───────────────────────────────────────
if [ "$WANT_MODELOPT" = 1 ]; then log "6a. modelopt env"; ensure_modelopt_env; fi
if [ "$WANT_VLLM" = 1 ];     then log "6b. vllm env";     ensure_vllm_env "$VLLM_VERSION"; fi

# ── 7. symlink farm (for the hardcoded-path eval_suite_*.sh) ─────────────────
if [ "$SYMLINK_FARM" = 1 ]; then
    log "7. symlink farm"
    # env alias: suite wants an env literally named 'omnimergekit'; alias $ENV_NAME.
    [ "$ENV_NAME" = omnimergekit ] || ln -sfn "$ENV_PFX" "$CONDA_ENVS/omnimergekit"
    mkdir -p /shared/dev
    ln -sfn "$REPO_DIR" /shared/dev/omnimergekit
    # WS path: /srv/.../backup_models -> /workspace. On a pod this path is pure
    # scaffolding (never user data), so replacing a stale real dir is safe.
    local_srv=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5
    mkdir -p "$local_srv"
    if [ -L "$local_srv/backup_models" ] || [ ! -e "$local_srv/backup_models" ]; then
        ln -sfn /workspace "$local_srv/backup_models"
    else
        log "   replacing stale real dir $local_srv/backup_models (pod scaffolding)"
        rm -rf "$local_srv/backup_models"; ln -sfn /workspace "$local_srv/backup_models"
    fi
    if [ -n "$TOKENIZER_LINK" ]; then
        tk_src="${TOKENIZER_LINK%%:*}"; tk_dst="${TOKENIZER_LINK##*:}"
        mkdir -p "$(dirname "$tk_dst")"
        ln -sfn "$tk_src" "$tk_dst"
        log "   tokenizer link $tk_dst -> $tk_src ($([ -e "$tk_dst/tokenizer.json" ] && echo OK || echo 'tokenizer.json MISSING'))"
    fi
fi

# ── 8. HF pulls ──────────────────────────────────────────────────────────────
if [ -n "$HF_PULL" ]; then
    export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported for --hf-pull}"
    HF_BIN="$ENV_PFX/bin/hf"
    [ -x "$HF_BIN" ] || HF_BIN="$(command -v hf || echo "$CONDA_ROOT/bin/hf")"
    log "8. HF pulls via $HF_BIN"
    "$HF_BIN" auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
    for pair in $HF_PULL; do
        repo="${pair%%:*}"; dir="${pair#*:}"
        if [ -f "$dir/model.safetensors.index.json" ] || [ -f "$dir/model.safetensors" ]; then
            log "   $dir present — skip"; continue
        fi
        log "   pull $repo -> $dir"
        # NB: modern `hf download` takes ONE pattern per --include (Click option,
        # not nargs) — repeat the flag per pattern (the 31M "DONE" trap, 2026-05-24).
        inc=(); for p in $HF_PULL_PATTERNS; do inc+=(--include "$p"); done
        HF_HUB_ENABLE_HF_TRANSFER=1 "$HF_BIN" download "$repo" --local-dir "$dir" "${inc[@]}" \
            || { log "FATAL pull $repo"; exit 1; }
    done
fi

# ── 9. verify ────────────────────────────────────────────────────────────────
log "9. verify"
ok=1
if [ "$DEPS" != none ] && [ "$DEPS" != train ]; then
    "$ENV_PY" -c "import lm_eval, transformers; print(f'  OK env: lm_eval={lm_eval.__version__} transformers={transformers.__version__}')" || ok=0
fi
if [ "$DEPS" = train ]; then
    "$ENV_PY" -c "import torch, transformers; print(f'  OK env: torch={torch.__version__} cuda={torch.cuda.is_available()} transformers={transformers.__version__}')" || ok=0
fi
if [ "$BUILD_LLAMA" = 1 ]; then
    for b in llama-server llama-quantize llama-imatrix; do
        test -x "/opt/llama.cpp/build/bin/$b" && echo "  OK   $b" || { echo "  MISS $b"; ok=0; }
    done
fi
if [ "$SYMLINK_FARM" = 1 ] && [ -n "$TOKENIZER_LINK" ]; then
    tk_dst="${TOKENIZER_LINK##*:}"
    test -e "$tk_dst/tokenizer.json" && echo "  OK   tokenizer reachable" || { echo "  MISS tokenizer.json"; ok=0; }
fi
[ "$ok" = 1 ] && log "BOOTSTRAP COMPLETE (GREEN)" || { log "BOOTSTRAP COMPLETE WITH ISSUES — see MISS above"; exit 1; }
