# omnimergekit eval — overview

This dir is the canonical home for all benchmark eval code. The v2 protocol
(2026-05-12 onward) is **vLLM-first, template-driven, scorer-validated,
token-stats-required**. Read `EVAL_PROTOCOL.md` (especially the v2 section
near the end) before any run.

## Quick start

```bash
# Run LCB-medium-55 against a local NVFP4A16 26B model
./omk_eval.py \
    --model /path/to/Gemma-4-26B-A4B-it-NVFP4A16 \
    --template lcb_medium_55 \
    --backend vllm \
    --quant auto

# Run HumanEval against a BF16 model (unquantized; preferred)
./omk_eval.py \
    --model /path/to/Qwen3.5-4B-MicroCoder \
    --template humaneval_full \
    --backend vllm \
    --quant bf16
```

`--template` accepts a bundled name (`humaneval_full`, `mbpp_full`,
`lcb_medium_55`, etc.) or a path to a custom YAML file. See
`templates/README.md` for the format.

## Files

| Path | Purpose |
|---|---|
| `omk_eval.py` | Unified runner — owns server lifecycle, eval dispatch, token-stats, sanity check |
| `templates/` | 9 canonical YAML templates + loader + spec |
| `validate_scorers.py` | Fixture-based scorer validation (per-bench, vs upstream lib) |
| `lcb/lcb_helpers.py` | Custom LCB shim (validated against official lcb_runner 13/13) |
| `lcb/lcb_llama_server.py` | Server runner the LCB shim uses |
| `tasks/` | lm-eval custom task YAMLs (smoke subsets HE-20, MBPP-40, GPQA-10/20) |
| `EVAL_PROTOCOL.md` | Mandatory checklist + v2 protocol section |

## Quant builder

Co-located at `../scripts/quantize_any.py`:

```bash
./scripts/quantize_any.py --src <bf16-dir> --dst <out-dir> --method nvfp4a16
./scripts/quantize_any.py --src <bf16-dir> --dst <out-dir> --method awq
./scripts/quantize_any.py --src <bf16-dir> --dst <out-dir> --method gptq
```

Each method runs in its dedicated env:
- `nvfp4a16` → `/root/anaconda3/envs/modelopt` (nvidia-modelopt 0.43.0)
- `awq`      → `/root/anaconda3/envs/vllm` (autoawq 0.2.9)
- `gptq`     → `/root/anaconda3/envs/modelopt` (gptqmodel 7.0, transformers 4.x)

## Validation status (2026-05-12)

| Bench | Scorer | Validated against | Cases |
|---|---|---|---:|
| HumanEval | exec | openai/human-eval | 3/3 |
| HumanEvalPlus | exec | evalplus.evaluate (extract-only) | 3/3 |
| MBPP | exec | exec assert in ns | 4/4 |
| GSM8K | flex-extract | lm-eval flexible-extract regex | 7/7 |
| AIME | strict extract | boxed + final-int regex | 6/6 |
| MMLU-Pro | flex-extract | lm-eval flexible-extract regex | 8/8 |
| GPQA | flex-extract | lm-eval flexible-extract regex | 6/6 |
| LCB-medium | exec | lcb_runner.evaluation.testing_util | 13/13 |

Re-run with: `./validate_scorers.py --bench all`

## Backend matrix

| Backend | When |
|---|---|
| vLLM (default) | BF16/FP16 if fits VRAM, else NVFP4A16/AWQ/GPTQ. New runs go here. |
| llama.cpp (`--backend llama`) | Existing GGUF tiers (Q2_K..Q6_K) we already built; Apple silicon. |
| opencoti-llamafile (`--backend llamafile`) | GGUF served by the opencoti-llamafile build, with optional lossless MTP speculative decoding. First-tier: auto binary (`--llamafile-bin`, env `OMK_LLAMAFILE_BIN`), `--mtp-head`/`--spec-n` for the draft-assistant, and think-then-answer (`deepseek` + budget) for code benches so weak/small models still emit clean fenced code. |
| Pod-side | Same `omk_eval.py` — pod is just a remote `--model` path. |

`--backend llamafile` example (MTP, code bench flags derived from the template):

```bash
omk_eval.py --backend llamafile --template humaneval_full \
    --model <f16.gguf> --tokenizer <hf-id> \
    --mtp-head <drafter.gguf> --spec-n 2
```

Unlike `--backend llama` (which uses `--reasoning off` for code benches — correct
on strong models), `llamafile` uses `--reasoning-format deepseek` +
`--reasoning-budget <template.thinking_token_budget>` for code benches too: a
weak model (e.g. Gemma-4 E2B) without the think scaffold answers coding prompts
with prose instead of a fenced function and pass@1 collapses (E2B HumanEval
0.6% → 83.5% with the scaffold). `--reasoning off` itself is not broken — it
correctly suppresses `<think>`; the model just needs the scaffold to emit code.

## Templates

```yaml
selection:
  type: indices   # frozen list — deterministic
  indices: [0, 13, 26, ...]
```
or
```yaml
selection:
  type: filter    # criteria — useful when dataset metadata defines the slice
  difficulty: medium
  min_date: "2024-10-01"
  testtype: functional
  doc_ids: ["abc", "def"]   # optional explicit allow list
```

The loader (`templates/template_loader.py`) refuses templates with
`n` ≠ `len(indices)` or missing required fields. Per-template YAMLs are
discoverable by name (`--template gsm8k_100`).

## Pre-flight checklist (before any run)

1. `vastai show instances` if pod-side
2. `nvidia-smi --query-gpu=memory.free` (≥ model size)
3. `df -h <results-dir>` (≥ 30 GB)
4. `./templates/template_loader.py <name>` (template loads)
5. `./validate_scorers.py --bench <name>` (scorer green if changed)

If anything fails, fix before launching. The protocol §v2.5 lists the
specific gotchas this catches.
