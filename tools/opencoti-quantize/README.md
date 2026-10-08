# opencoti-quantize — a `llama-quantize` that writes PQ2_0 and PTQ1_0

`llama-quantize` from opencoti, our llamafile / llama.cpp fork (not public yet), as of its patch `0590-bonsai2-types` (c11 row 3.1). On top of every
stock type it writes the two Ternary Bonsai 2 weight types, at the ids PrismML's published GGUFs carry:

| type | ftype | ggml id | what |
|---|---|---|---|
| `PQ2_0` | 141 | 142 | the prism fork's Q2_0 codec at group 128 — fp16 scale + 32 bytes of 2-bit codes, 34 bytes / 128 weights (2.13 bpw) |
| `PTQ1_0` | 143 | 143 | ternary at group 128 — 24 + 2 bytes of base-3 packed trits + fp16 scale, 28 bytes / 128 weights (1.75 bpw); upstream llama.cpp PR #29672's code, verbatim |

Both are meant for **ternary-trained** weights (Bonsai-style quantization-aware training). Quantizing an ordinary model to them is
valid and runs, but it is a heavy loss: these are not general low-bit formats.

## Use

```sh
bin/opencoti-quantize model-F16.gguf model-PTQ1_0.gguf PTQ1_0 [threads]
bin/opencoti-quantize model-F16.gguf model-PQ2_0.gguf  PQ2_0  [threads]
```

Same options as `llama-quantize` (`--help`). The binary is an APE (Cosmopolitan): one file runs on Linux, Windows and macOS x86_64.
It is CPU-only and needs no GPU library.

## Verified (2026-10-08)

- **Same output as PrismML's own tool.** LFM2.5-1.2B F16 → PTQ1_0 and → PQ2_0 with this binary and with `llama-quantize` from
  `prism-b10754`: every tensor's type, shape and data bytes identical (62 tensors, 38 quantized, both types).
- **Same codec as the published files.** For six tensors of each published `Ternary-Bonsai-2-27B` file (22 200 320 blocks each),
  dequantize → requantize reproduces the published bytes exactly; ours, the fork's C code and gguf-py dequantize identically.
- Details: in the opencoti tree, `vendors/patches/llamafile/README.md`, section `0590-bonsai2-types`.

## The binary

`bin/` is not in git (this repo keeps code only). `SHA256SUMS` names the build this README describes:

- source: opencoti `dev` at `9658a3f530` (patch chain through `0590`), the build tree of engine build `2610081309001`;
- rebuild: `./build.sh <opencoti build tree>/vendors/sources/llamafile bin/opencoti-quantize` — it links the objects the opencoti
  build already compiled (the pipeline compiles `quantize.cpp` but links no quantize binary). Reproducible: a rebuild from the
  same tree gives the same sha256.

## Running the result

A PQ2_0 / PTQ1_0 GGUF runs on PrismML's llama.cpp fork. In opencoti the types load from c11 row 3.1; the CPU kernels are row 3.2
and the CUDA / Vulkan kernels rows 3.3 / 3.7 — until a backend has a kernel the engine refuses the model by name
(`type pq2_0: no backend in this build has a kernel for it`) rather than running it.
