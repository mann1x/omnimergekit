#!/usr/bin/env python3
"""Create + populate ManniX-ITA/Qwen3.6-27B-A3B-CoderX (bf16 safetensors) from armJ.

Staging is HARDLINKS, not copies: the shards are 51 GB and /mnt/sdc is the only fs with
room. Hardlinking also means the uploaded bytes are provably the same inodes the shipped
GGUFs were converted from -- no re-derivation, no chance of uploading a different build.

The ONE edit vs the on-disk build is model.safetensors.index.json's `total_size`, which
omits mtp.safetensors entirely (51,190,031,616 recorded vs 52,426,033,408 of real BF16
tensor bytes -- a 1.236 GB shortfall exactly equal to the grafted MTP block, because the
index was written before the R1.mtp graft). Tensor data is untouched; weight_map already
maps mtp.* -> mtp.safetensors correctly.

Repo is created PRIVATE. Flipping it public is a separate, deliberate act.
"""
import json
import os
import pathlib
import shutil
import sys

from huggingface_hub import HfApi

SRC = pathlib.Path("/mnt/sdc/ream-work/armJ")
STAGE = pathlib.Path("/mnt/sdc/ream-work/coderx_st_upload")
REPO = "ManniX-ITA/Qwen3.6-27B-A3B-CoderX"
TRUE_TOTAL = 52_426_033_408          # recomputed from every shard header, all BF16

README = """---
base_model: Qwen/Qwen3.6-35B-A3B
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - moe
  - expert-pruning
  - code
  - mtp
  - omnimergekit
---

# Qwen3.6-27B-A3B-CoderX

BF16 weights for **CoderX** — a long-horizon code prune of
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B): 256 experts per layer
reduced to **184**, ~35B → ~27B, still A3B active.

Same expert budget as the sibling
[Qwen3.6-27B-A3B-Coder](https://huggingface.co/ManniX-ITA/Qwen3.6-27B-A3B-Coder), but a
different **selection** plus a **redistribution** step: our saliency map picks the keep-set,
a REAP-style per-layer floor (p=24) protects the tail, and the 72 evicted experts per layer
are folded DERN-style into the survivors rather than discarded. Router, attention and norms
are otherwise untouched. No fine-tuning, no distillation.

Built with [omnimergekit](https://github.com/mann1x/omnimergekit).

## Read this before you load it

- **Routing is top-8** (`num_experts_per_tok: 8`) — the base model's native setting, and
  measured rather than assumed: MBPP-full 0.784 / 0.790 at top-8 against 0.732 / 0.730 at
  top-10. The Coder sibling bakes top-10; this one does not.
- **The MTP block is included** (`mtp.*`, in `mtp.safetensors`, indexed) for speculative
  decoding.
- **This checkpoint is text-only.** `architectures: Qwen3_5MoeForCausalLM`,
  `model_type: qwen3_5_moe_text` — the base model's vision tower did **not** survive the
  redistribution step and is not present here. The Coder sibling's safetensors repo *is*
  multimodal (`Qwen3_5MoeForConditionalGeneration`); this one is not. The `vision-<tier>`
  tags on ollama get their vision tower from the Coder mmproj at the GGUF layer, so they
  are unaffected — but if you need a multimodal **safetensors** checkpoint, use Coder.

## Quantised builds

- GGUF (19 tiers, all imatrix, MTP included):
  [`Qwen3.6-27B-A3B-CoderX-MTP-GGUF`](https://huggingface.co/ManniX-ITA/Qwen3.6-27B-A3B-CoderX-MTP-GGUF)
- ollama: `ollama run mannix/qwen3.6-27b-a3b-coderx`

## Evaluation

Q6_K + imatrix, llama.cpp b9700, **greedy** (`temperature 0.0 / top_p 1.0 / top_k 0`), one
pinned serving geometry per bench, read back from the server log.

| Benchmark | **CoderX** | Coder (184e) | Qwen3.6-35B-A3B (256e) |
|---|---|---|---|
| LiveCodeBench v6 (77q, 24k think / 48k total) | **0.727** | 0.610 | 0.610 |
| HumanEval+ (164) | **0.970** | 0.951 | 0.939 |
| MultiPL-E-100 (rs+java+js, 300 completions) | 0.887 | 0.890 | 0.910 |

A same-basis repeat of MultiPL-E moved 1.0 pp on batch-scheduling nondeterminism alone, so
the 0.33 pp CoderX↔Coder gap is a **tie**; the 2.33 pp gap to the teacher is real. Per
language (CoderX / Coder / 256e): Rust **0.85** / 0.81 / 0.84 · Java 0.89 / 0.90 / 0.93 ·
JS 0.92 / 0.96 / 0.96.

The full canonical 9-bench suite has **not** been run on this checkpoint yet; non-code axes
(GPQA, MATH-500, IFEval, ARC) are deliberately not quoted here.

Apache-2.0 · research checkpoint.
"""


def main() -> int:
    api = HfApi()
    print("whoami:", api.whoami()["name"])

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    idx = json.loads((SRC / "model.safetensors.index.json").read_text())
    old = idx["metadata"]["total_size"]
    files = sorted(set(idx["weight_map"].values()))

    # Hardlink every referenced shard; refuse if any is absent rather than uploading a
    # partial checkpoint that would fail to load only at the consumer's end.
    for f in files:
        src = SRC / f
        if not src.exists():
            print(f"REFUSE: index references {f} but it is not on disk")
            return 2
        os.link(src, STAGE / f)

    idx["metadata"]["total_size"] = TRUE_TOTAL
    (STAGE / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))
    print(f"index total_size {old} -> {TRUE_TOTAL}  (+{TRUE_TOTAL - old})")

    for f in ("config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy2(SRC / f, STAGE / f)
    (STAGE / "README.md").write_text(README)

    staged = sorted(p.name for p in STAGE.iterdir())
    print(f"staged {len(staged)} files:", [s for s in staged if not s.endswith('.safetensors')])

    # A sanity floor: the shards alone must exceed 50 GB, else something hardlinked wrong.
    total = sum((STAGE / f).stat().st_size for f in files)
    if total < 50e9:
        print(f"REFUSE: staged shards total only {total/1e9:.1f} GB")
        return 3

    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    print(f"repo ready (PRIVATE): https://huggingface.co/{REPO}")

    api.upload_large_folder(repo_id=REPO, repo_type="model", folder_path=str(STAGE),
                            num_workers=4)
    print("UPLOAD_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
