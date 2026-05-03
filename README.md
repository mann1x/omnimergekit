# omnimergekit

Tools for **model merging**, **expert pruning**, **differential competence-map extraction**, and **GGUF quantization** — built on top of `transformers`, `safetensors`, and `llama.cpp`. The kit started as a fork-flavored alternative to mergekit (`omnimerge_v2` recipe) and grew to cover Gemma 4 MoE surgery and Qwen3.5 hybrid-attention frankenmerges.

> **Status:** research code. Not packaged for pip yet. Scripts assume the directory layout described below; paths inside scripts may need editing for your environment.

## What's inside

| Path | Purpose |
|------|---------|
| `omnimergekit.py` | The main merge script. Methods: `dare_ties`, `omnimerge_v2`, plus features `obim`, `darex`, `emr`, `fisher`. |
| `competence/` | Differential competence-map pipeline: extract per-source per-task fisher signal → combine across tasks → feed into `omnimergekit.py --fisher`. |
| `gemma4/` | Gemma 4 MoE surgery: expert drop, DERN-style redistribution, CD-maps for contribution-aware quants, hybrid-expert assembly. |
| `quantization/` | `quantize_gguf.py` (multi-tier quants with imatrix), `convert_to_4bit.py`, `publish_model.py` (HF push with frontmatter). |
| `eval/` | Eval drivers — GPQA Diamond, LiveCodeBench (lcb_llama_server.py), HE/MBPP rescore-with-fence-strip helpers. |
| `recipes/` | End-to-end pipelines: 4B MicroCoder series, Gemma 4 109e/98e/120e/128e, 27B Omnimerge. |
| `pod/` | RunPod / Vast.ai helpers (setup, parallel run, retrieve, README publish). |
| `docs/` | Method docs, experiment journals, recipe deep-dives. |
| `experiments/` | Per-experiment notebooks / logs. |

## Two recipes you'll actually use

### 1. Frankenmerge with Fisher importance

```bash
# 1. Run HE / MBPP / etc. eval on each source model, save lm-eval samples_*.jsonl
# 2. Extract per-source fisher signal restricted to docs THAT source uniquely solved
python competence/competence_extract.py \
    --model $SRC1 --samples $EVAL/source1/samples_humaneval_*.jsonl \
    --task he --keep-doc-ids 1,4,7,11 \
    --output maps/source1__he.safetensors \
    --max-len 4096 --chunk-len 1280     # chunked grad-accum: full context on small VRAM

# 3. Combine across tasks (per source) into single competence map
python competence/competence_combine.py \
    --map "source1:humaneval:results.json:maps/source1__he.safetensors" \
    --map "source1:mbpp:results.json:maps/source1__mbpp.safetensors" \
    --raw-rate --signal weight_taylor \
    --output-dir maps/combined/

# 4. Merge with Fisher-aware omnimerge_v2
python omnimergekit.py \
    --base $BASE --source $SRC1 --source $SRC2 --source $SRC3 \
    --output merged/ \
    --method omnimerge_v2 --v2-features fisher,darex \
    --weights 0.35,0.40,0.25 --density 0.53 --darex-q 0.85 \
    --fisher "maps/combined/source1.safetensors,maps/combined/source2.safetensors,maps/combined/source3.safetensors" \
    --pr682-turbo --skip-patterns "model.visual,mtp.layers" --device cuda
```

See [`docs/METHOD_omnimerge_v2.md`](docs/METHOD_omnimerge_v2.md) for the math (OBIM-lite + DAREx-q + EMR election + Fisher).

### 2. Gemma 4 MoE expert drop with router recalibration

```bash
# Drop weakest 19 experts/layer (128e → 109e), preserving routing semantics
python gemma4/expert_pruning/expert_drop.py \
    --model gemma-4-26B-A4B-it --n-keep 109 \
    --analysis gemma4/neuron_analysis/expert_neuron_v4.json \
    --output gemma-4-A4B-109e/ --recalibrate-router 2000

# Quantize with contribution-aware (CD) maps
python gemma4/cd_maps/generate_cd_maps_from_contribution.py \
    --analysis gemma4/neuron_analysis/expert_neuron_v4.json \
    --output cd_maps/

python quantization/quantize_gguf.py gemma-4-A4B-109e \
    --tier CD-Q4_K_M --cd-maps cd_maps/ \
    --imatrix calibration.dat --out gemma-4-A4B-109e-CD-Q4_K_M.gguf
```

See [`docs/METHOD_gemma4_pruning.md`](docs/METHOD_gemma4_pruning.md) for full method.

## Hard-won rules

These are baked into the recipe scripts. They cost real compute or a published-model rollback to learn.

- **Always `--use_cache <path>` and `--log_samples` on lm_eval.** Without sqlite cache, any death (PEG parser, OOM, network blip) restarts from 0. Without samples, you can't tell whether `pass@1=0` means "model bad" or "scorer crashed on markdown fences".
- **`imatrix.dat` MUST be archived next to every quant.** Recompute is 15-20 min of GPU time and depends on calibration data + seed. Lose it → quant cannot be reproduced bit-for-bit.
- **Never run `lm_eval` on a chat model via `/v1/completions` without an explicit chat template.** Gemma 4 / Qwen3.5 reasoning variants emit fenced code that scorers can't `exec()`. Use `/v1/chat/completions + apply_chat_template`, or rescore samples with the fence-strip helpers in `eval/`.
- **Gemma 4 needs `--reasoning-format deepseek --reasoning-budget 8192`** when served via `llama-server`. Without budget, it emits malformed channel tokens and crashes eval mid-run.
- **Qwen3.5 has hybrid linear/full attention.** Without `flash-linear-attention` and `causal-conv1d` installed, gradient extraction OOMs at >1k context. Either install them or use chunked-grad-accum (`competence_extract.py --chunk-len`).

## Published artifacts using this kit

- [`ManniX-ITA/Qwen3.5-27B-Omnimerge-v2`](https://huggingface.co/ManniX-ITA/Qwen3.5-27B-Omnimerge-v2) — 27B frankenmerge (4 sources, OBIM-lite + DAREx-q + EMR + Fisher).
- [`ManniX-ITA/Qwen3.6-27B-Omnimerge-v3a`](https://huggingface.co/ManniX-ITA/Qwen3.6-27B-Omnimerge-v3a) — cross-base v3a (Qwen3.6 base + 3 Qwen3.5 sources).
- [`ManniX-ITA/Qwen3.6-27B-Omnimerge-v3b`](https://huggingface.co/ManniX-ITA/Qwen3.6-27B-Omnimerge-v3b) — same-base v3b.
- Gemma 4 A4B 109e — pruned MoE (128e → 109e), 75.25% → 71.72% GPQA Diamond, ~12 GB Q4_K_M.

## License

MIT (see `LICENSE`).

## Citation

If you use this in published work:
```
@misc{omnimergekit,
  author = {Calpini, Federico},
  title = {omnimergekit: model merging, expert pruning, and differential competence maps},
  year = {2026},
  howpublished = {\url{https://github.com/mann1x/omnimergekit}}
}
```
