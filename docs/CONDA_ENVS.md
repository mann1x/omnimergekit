# Conda environments

omnimergekit ships with a single conda env definition that covers merging,
competence-map extraction, evaluation, and quantization helpers. Specialized
envs (e.g. `omnimergekit-gemma4`) only exist when an architecture forces a
different transformers / torch combination than the main env can provide.

## Quick start

```bash
# main env (Python 3.11)
conda create -n omnimergekit python=3.11 -y
conda activate omnimergekit

pip install -r requirements.txt
pip install -r requirements-eval.txt    # if you'll run lm-eval
pip install -r requirements-quant.txt   # if you'll quantize GGUFs
pip install -r requirements-dev.txt     # optional: linters, jupyter, plots
```

> **Python 3.11 is the default.** If a pinned wheel doesn't exist for 3.11 on
> your platform (rare), fall back to `python=3.10`. The lightseek venv that
> seeded these versions runs on 3.10.18 and works identically; nothing in the
> stack is 3.11-only.

### Env activation hooks (recommended)

Add to `~/.bashrc` or per-project `.envrc`:

```bash
alias omk='conda activate omnimergekit && cd /path/to/omnimergekit'
export HF_ALLOW_CODE_EVAL=1
export PYTHONDONTWRITEBYTECODE=1   # kills stale .pyc files that bite on lm_eval upgrades
```

## Why these versions

| Pin | Why |
|-----|-----|
| `python==3.11` | Long-tail wheels available; 3.13 has not been validated against `flash-linear-attention` and several CUDA kernels. |
| `torch==2.10.0` | Validated with `transformers==5.5.0`. SDPA + bf16 grad-checkpoint paths are stable. |
| `transformers==5.5.0` | **Required** for Qwen3.5 / Qwen3.6 hybrid-attention models AND Gemma 4 26B-A4B. Earlier 4.x versions miss the model classes; later 6.x has API breaks we haven't ported. |
| `flash-linear-attention>=0.5.0` + `causal-conv1d>=1.6.0` | Without these, Qwen3.5 falls back to a torch attention impl that materializes huge per-layer state and OOMs at >1k context during gradient extraction. **Hard requirement** for `competence_extract.py` on Qwen3.5 / Qwen3.6. |
| `safetensors==0.7.0` | The 0.5/0.6 line had streaming-load bugs we hit with multi-shard 27B BF16 frankenmerges. |
| `accelerate==1.13.0` | Matches `device_map="auto"` semantics expected by `quantize_gguf.py`'s ImatrixCompute path. |
| `bitsandbytes==0.49.2` | Optional. Used only by `quantization/convert_to_4bit.py`. |
| `lm-eval[api]==0.4.11` | The `[api]` extras bring `tenacity` + `requests` which the `local-completions` model backend needs at construction time. Bare `pip install lm-eval` is missing them and dies. |
| `huggingface_hub>=0.27.0` | The `hf download/upload/auth login` CLI replaces the deprecated `huggingface-cli` (which silently exits 0 with no transfer on >0.27). |

## Architecture-specific envs

> **Most users do not need these.** The main `omnimergekit` env covers Qwen3.5,
> Qwen3.6, and Gemma 4. Create a specialized env only if you hit a concrete
> incompatibility.

### `omnimergekit-gemma4` (placeholder)

Currently identical to the main env — Gemma 4 26B-A4B works on
`transformers==5.5.0` + `torch==2.10.0`. Reserved for future use if Google
ships an updated Gemma 4 architecture that requires a different transformers
release. If you create it:

```bash
conda create -n omnimergekit-gemma4 python=3.11 -y
conda activate omnimergekit-gemma4
pip install -r requirements.txt -r requirements-eval.txt -r requirements-quant.txt
# diverge from here as needed
```

When you diverge, copy `requirements.txt` → `requirements-gemma4.txt` and
edit the pin that needs to change. Do **not** edit the shared
`requirements.txt`.

## Building llama.cpp (external)

The kit assumes `llama.cpp` binaries are at `/opt/llama.cpp/build/bin/`. To
build:

```bash
cd /opt && git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16
```

The Python helper `convert_hf_to_gguf.py` lives at `/opt/llama.cpp/convert_hf_to_gguf.py`
and runs in **the same conda env as omnimergekit** — it imports `transformers`
and the same model classes, so do not run it from a separate env unless you
mirror the version pins.

### imatrix calibration archival rule

Every `imatrix.dat` used to build a quant **must** be saved next to the
quants and uploaded to the matching HF repo. Recompute is 15-20 min of
GPU time, depends on calibration data + seed + ngl, and cannot be
reproduced bit-for-bit if lost. See top-level README "Hard-won rules".

## Reproducing the lightseek venv

If you need to bootstrap a non-conda venv that matches what the recipe
scripts originally ran on:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-eval.txt -r requirements-quant.txt
```

The `recipes/` shell scripts hardcode `PYBIN=/shared/dev/lightseek/.venv/bin/python`
in many places. After migration to the conda env, update those paths or
symlink:

```bash
ln -s $CONDA_PREFIX /shared/dev/lightseek/.venv   # only if you replace lightseek entirely
```
