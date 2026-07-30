#!/usr/bin/env python3
"""
Pre-quantize a model to 4-bit NF4 and save to disk.
This avoids loading bf16 weights into VRAM during inference.
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import torch

os.environ["HF_TOKEN"] = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

# NOTE: the three imports below are deliberately NOT at the top of the file, and the two
# `noqa: E402` marks are load-bearing, not cosmetic. Order of operations matters:
#   1. HF_TOKEN must be in os.environ before anything touches huggingface_hub;
#   2. bnb.nn.Params4bit.__new__ must be patched before transformers constructs any
#      Params4bit (it passes _is_hf_initialized, which upstream bnb rejects);
#   3. caching_allocator_warmup must be neutered before the model is loaded, or the
#      warmup allocates the full bf16 footprint in VRAM — the exact thing this script
#      exists to avoid.
# Moving them to the top silently breaks 4-bit conversion. Do not 'clean this up'.
import bitsandbytes as bnb
_orig = bnb.nn.Params4bit.__new__
def _p(cls, *a, **k):
    k.pop("_is_hf_initialized", None)
    return _orig(cls, *a, **k)
bnb.nn.Params4bit.__new__ = _p

import transformers.modeling_utils as _mu  # noqa: E402  (see note above)
_mu.caching_allocator_warmup = lambda *a, **k: None

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {source} in 4-bit on CPU...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(source),
        quantization_config=bnb_config,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    print(f"Loaded in {time.time()-t0:.0f}s")

    print(f"Saving 4-bit model to {output}...")
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    # Copy extra files
    for fn in ["chat_template.jinja", "processor_config.json",
               "preprocessor_config.json", "expert_drop_metadata.json"]:
        src = source / fn
        if src.exists():
            shutil.copy2(src, output / fn)

    print(f"Done! Saved to {output}")
    print(f"Size: {sum(f.stat().st_size for f in output.rglob('*.safetensors')) / 1024**3:.1f} GB")


if __name__ == "__main__":
    main()
