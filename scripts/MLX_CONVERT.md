# MLX Conversion — `mlx_convert.sh`

End-to-end pipeline for converting a Hugging Face model to MLX-quantized
format (4-bit, 8-bit, or bf16) and uploading the result back to Hugging Face.

This script lives next to the other OmniMergeKit pod helpers and is the
**canonical** way to produce MLX repos for any project. Do not write one-off
MLX-conversion shells; extend this one.

## What it produces

A new Hugging Face repo (created if missing) containing:

```
config.json            ← includes quantization: {group_size: 64, bits: N}
model.safetensors      ← quantized weights (sharded if large)
chat_template.jinja    ← copied from source tokenizer
tokenizer.json
tokenizer_config.json
generation_config.json
README.md              ← minimal MLX card (overwrite with a richer one if you want)
```

End users (typically on Apple Silicon) load it with:

```python
from mlx_lm import load, generate
m, t = load("ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-MLX-4bit")
print(generate(m, t, prompt="...", max_tokens=200, verbose=True))
```

## Where it runs

- **Apple Silicon Mac**: works natively with the unmodified `mlx` + `mlx-lm`
  packages from PyPI. Skip the conda env block.
- **Linux + NVIDIA GPU** (vast.ai RTX 3090, your home box, etc.): works via the
  `mlx-cuda` backend. The script auto-creates an isolated conda env (`mlx`) with
  the known-working version pin so it doesn't collide with your pytorch env.
- **Linux + CPU only**: not supported; mlx-cuda requires CUDA and the CPU-only
  wheel for mlx 0.30+ ships an empty `libmlx.so` placeholder that fails to load.

## Why these specific versions

```
mlx==0.30.0
mlx-cuda==0.30.0      ← MUST match mlx exactly (ABI-coupled)
mlx-lm==0.30.7        ← lower bound for Qwen3.5/3.6 (qwen3_5.py first
                         shipped in 0.30.0). Older mlx-lm doesn't know
                         model_type=qwen3_5 and rejects the convert.
```

Note: a prior version of this doc pinned `mlx-lm<0.27` because mlx-lm 0.27+
imports `mx.new_thread_local_stream` (absent in mlx-cuda 0.30). In practice
that symbol is only touched at **inference** time — `mlx_lm.convert` runs
cleanly with mlx-lm 0.30.7 on mlx-cuda 0.30. Confirmed 2026-05-11 on
Qwen3.6-27B-Omnimerge-v4 → 15 GB MLX-4bit shards in ~2 min.

This is the only known-working combination on Linux+CUDA as of 2026-05.
**Do not bump any of these in isolation** — bump all three together and
re-run a smoke convert before propagating. The mlx and mlx-cuda packages on
PyPI track each other; mlx-lm trails by a few versions.

A side effect of installing `mlx-cuda` is that it pulls a newer
`nvidia-cublas-cu12 (12.9)` and `nvidia-cuda-nvrtc-cu12 (12.9)`, which
**conflict with pytorch 2.5.1+cu121** (wants 12.1.x). Keep MLX in its own
conda env to avoid breaking torch.

Inference on the CUDA build currently fails with a `cooperative_groups` CUDA
SDK mismatch (`cudaCGGetRank undefined`). This affects **only** running the
model on the pod; conversion + upload works fine, and end users on Apple
Silicon use the native mlx runtime which doesn't have this issue.

## Usage

```bash
# 4-bit (most common)
HF_TOKEN=$(cat ~/.cache/huggingface/token) bash mlx_convert.sh \
    ManniX-ITA/Qwen3.6-27B-Omnimerge-v4 \
    ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-MLX-4bit \
    4

# 8-bit (better quality, ~2× size of 4-bit)
HF_TOKEN=$(cat ~/.cache/huggingface/token) bash mlx_convert.sh \
    ManniX-ITA/Qwen3.6-27B-Omnimerge-v4 \
    ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-MLX-8bit \
    8

# bf16 (unquantized; same precision as source)
HF_TOKEN=$(cat ~/.cache/huggingface/token) bash mlx_convert.sh \
    ManniX-ITA/Qwen3.6-27B-Omnimerge-v4 \
    ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-MLX-bf16 \
    16

# Custom group size, e.g. for 4-bit/g128 (less aggressive than default g64):
HF_TOKEN=$(cat ~/.cache/huggingface/token) bash mlx_convert.sh \
    ManniX-ITA/Qwen3.6-27B-Omnimerge-v4 \
    ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-MLX-4bit-g128 \
    4 --q-group-size 128
```

The third arg controls the bit width:
- `4` → emits `-q --q-bits 4` to `mlx_lm.convert` (default group_size 64)
- `8` → emits `-q --q-bits 8`
- anything else → no `-q`; conversion preserves source dtype (typically bf16)

Any additional args after the bit-width are passed straight to `mlx_lm.convert`
(see `mlx_lm.convert --help` for the full set).

## Required environment

- `HF_TOKEN` — write token for the target repo. Read token won't work
  because the script also creates the target repo if it doesn't exist.

## Optional environment

- `WORKDIR` — where to stage downloads and outputs. Default
  `/workspace/mlx_convert`. Each (source, target) pair gets its own
  subdirectory so concurrent conversions don't collide.
- `ENV_NAME` — conda env name. Default `mlx`.
- `CONDA_ROOT` — conda installation root. Default `/opt/conda`.

## Disk budget

The script downloads the source weights to `$WORKDIR/<target>/source` (~size
of the unquantized model in bf16), then writes the converted weights to
`$WORKDIR/<target>/mlx` (~bits/16 × source size). For a 27B bf16 model
converted to 4-bit:

- source download: ~54 GB
- mlx-4bit output: ~14 GB
- transient hf-transfer buffer: ~5 GB
- **total peak: ~75 GB**

For an 8-bit conversion bump the output to ~27 GB; total peak ~85 GB.

Plan disk accordingly. The script does not delete the source after the run
(in case you want a second 8-bit pass); clean up with
`rm -rf $WORKDIR/<target>/source` once you're done.

## Troubleshooting

- **`ImportError: libmlx.so: cannot open shared object file`** —
  `mlx-cuda` wasn't installed. The pure `mlx` wheel on Linux is a stub.
  Verify the conda env was activated and `mlx-cuda` is present:
  `pip show mlx-cuda`.

- **`AttributeError: module 'mlx.core' has no attribute 'new_thread_local_stream'`** —
  `mlx-lm` is too new for the pinned `mlx` version. Pin `mlx-lm<0.27`.

- **CUDA `cudnnBackendPopulateCudaGraph` undefined symbol** —
  Pod's cudnn is too old for the `mlx-cuda` build. Try
  `pip install -U nvidia-cudnn-cu12`. If that doesn't help, the mlx-cuda
  wheel doesn't ship its own cudnn and relies on the system one; the
  vast.ai pytorch image's cudnn 9.1 sometimes diverges from what mlx expects.

- **Inference fails with `cooperative_groups/helpers.h ... cudaCGGetRank
  undefined`** — this is expected on Linux + mlx-cuda 0.30. Conversion
  still works. Don't try to run the model on the pod; download to Apple
  Silicon to test.

## Related

- `omnimergekit/scripts/quantize_gguf.py` — GGUF quantization pipeline
  (different artifact: llama.cpp Q4_K_M, IQ4_XS, CD-* quants, etc.)
- `omnimergekit/eval/EVAL_PROTOCOL.md` — locked methodology for evaluating
  the produced quants.

## Version history

- **2026-05-10** — Initial. Pinned `mlx==0.30.0` + `mlx-cuda==0.30.0` +
  `mlx-lm<0.27` after a long debugging session involving 3 wrong combinations
  of mlx wheels on a vast.ai pod.
