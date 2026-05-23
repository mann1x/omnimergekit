# Ollama tooling

Utility scripts under `scripts/` for publishing to and inspecting models on
ollama.com.

## scripts/ollama_push_gemma4.sh / ollama_push_generic.sh

Push pre-quantized GGUFs (multi-tier) from local disk to ollama.com under
`mannix/<model>` with the canonical Gemma 4 renderer + parser. Both scripts
share the same flags:

```bash
bash scripts/ollama_push_gemma4.sh \
  --gguf-dir <dir-with-Q*_K_*.gguf-files> \
  --target mannix/gemma4-98e-v5-coder \
  --latest-tier Q4_K_M   # tag this tier as :latest in addition to its own tag
```

The `--latest-tier` default is `Q4_K_M`. Use `--no-latest` only for partial-tier
re-runs (CD-* rebuild) that must not clobber the existing `:latest`. The script
pre-flight bails if the requested latest-tier isn't in the planned tier set
AND isn't already on ollama.com.

## scripts/ollama_backfill_latest.sh

One-shot fix for models already published without a `:latest` tag.
`pull → cp → push → sweep orphan blobs`. Idempotent — skips models that
already have `:latest` on ollama.com.

```bash
bash scripts/ollama_backfill_latest.sh mannix/gemma4-98e-v5-coder Q4_K_M
```

## scripts/ollama_inspect_model.py

Inspects any Ollama-published model from the public registry. Returns:

- **Config blob** — `model_family`, `model_type`, `file_type` (quant),
  `renderer`, `parser`
- **Layer manifest** from `registry.ollama.ai/v2/` — list of layer mediaTypes
  with digests and sizes (`.model` GGUF, `.template`, `.params`, `.system`,
  `.license`, `.adapter`, `.projector`)
- **Runtime params** — `num_ctx`, `temperature`, `top_p`, `stop`, etc.
- **GGUF metadata table** — `general.architecture`, `<arch>.block_count`,
  `<arch>.embedding_length`, `<arch>.attention.head_count`,
  `<arch>.context_length`, `<arch>.rope.*`, `tokenizer.ggml.*`,
  `quantize.imatrix.*`

### Why two modes

The Docker-compatible registry at `registry.ollama.ai/v2/` only exposes raw
blobs — there's no JSON endpoint that returns parsed GGUF metadata. The
ollama.com web frontend is a separate service that parses GGUF headers
server-side and renders the metadata table into HTML. The default
`--mode scrape` reads that HTML to get the parsed view without downloading
the multi-GB GGUF blob. `--mode registry` skips the scrape and reports only
manifest + config blob (faster, smaller surface, more robust).

### Usage

```bash
# Pretty render
python3 scripts/ollama_inspect_model.py mannix/gemma4-31b-he1:IQ2_S

# JSON (for piping into jq / a watchdog)
python3 scripts/ollama_inspect_model.py mannix/gemma4-31b-he1:IQ2_S --json

# Manifest + config only (zero scraping)
python3 scripts/ollama_inspect_model.py mannix/gemma4-31b-he1:IQ2_S --mode registry
```

### Example output (mannix/gemma4-31b-he1:IQ2_S)

```
=== mannix/gemma4-31b-he1:IQ2_S ===
  family       gemma4
  type         30.7B
  quant        unknown
  renderer     gemma4
  parser       gemma4

=== Layers ===
  model        sha256:230aab877f4d...      9.46 GiB

=== GGUF metadata ===
  general.architecture                                    gemma4
  general.file_type                                       IQ2_S
  gemma4.attention.head_count                             32
  gemma4.attention.head_count_kv                          [16, 16, 16, 16, 16, ...]
  gemma4.block_count                                      60
  gemma4.context_length                                   262144
  gemma4.embedding_length                                 5376
  gemma4.feed_forward_length                              21504
  gemma4.final_logit_softcapping                          30
  gemma4.rope.freq_base                                   1e+06
  ...
  quantize.imatrix.chunks_count                           128
  quantize.imatrix.dataset                                /workspace/calibration_datav5.txt
  quantize.imatrix.entries_count                          410
```

### Caveats

- **Tag enumeration**: `registry.ollama.ai/v2/<ns>/<model>/tags/list` returns
  404. Scrape `https://ollama.com/<ns>/<model>/tags` (HTML) instead.
- **Scrape brittleness**: the parser keys on ollama.com's HTML class names
  (`text-neutral-600 sm:text-black`). UI refactors can break the regex; the
  robust fallback is registry v2 + local `gguf-py` parse of a range-pulled
  blob header (~1 MB).
- **Zero deps**: the script uses only Python stdlib — no `requests`, no
  `gguf-py`. Works on any host that can reach ollama.com / registry.ollama.ai.
