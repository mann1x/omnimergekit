#!/usr/bin/env python3
"""Rename keys in lrp_scores.safetensors to match the multimodal-wrapped
safetensors index (model.language_model.* prefix).

When `Qwen3_5ForConditionalGeneration.from_pretrained(...)` is loaded and
then `named_parameters()` is called, pytorch returns shorter paths
(`model.layers.X.*`) because the inner LM module is the `model.language_model`
submodule of the wrapper. But the safetensors index — which mergekit reads —
uses the full multimodal path (`model.language_model.layers.X.*`).

This script bridges that gap by adding the `language_model.` prefix to
keys that need it. Used by the M4 ex-LRP pipeline.
"""
import argparse
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


def remap_key(k: str, lang_prefix: str = "language_model.", reverse: bool = False) -> str:
    """Add or remove the language_model. prefix.

    Forward (default):
    - `model.<x>` → `model.language_model.<x>` (unless already prefixed)
    - `lm_head.weight` stays at root

    Reverse (--reverse):
    - `model.language_model.<x>` → `model.<x>`
    """
    if reverse:
        if k.startswith("model." + lang_prefix):
            return "model." + k[len("model." + lang_prefix) :]
        return k
    if k.startswith("model.") and not k.startswith("model.language_model."):
        return "model." + lang_prefix + k[len("model.") :]
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="lrp_scores.safetensors")
    ap.add_argument("output", type=Path, help="output safetensors path")
    ap.add_argument("--inplace", action="store_true",
                    help="overwrite input (write to <output>.tmp then mv)")
    ap.add_argument("--drop-keys", default="",
                    help="comma-separated key prefixes to drop entirely")
    ap.add_argument("--reverse", action="store_true",
                    help="reverse direction: strip the language_model. prefix instead of adding it")
    args = ap.parse_args()

    drop = [k.strip() for k in args.drop_keys.split(",") if k.strip()]

    print(f"loading {args.input}", flush=True)
    scores = load_file(str(args.input))
    print(f"  {len(scores)} keys loaded", flush=True)

    out = {}
    n_remapped = 0
    n_dropped = 0
    for k, v in scores.items():
        if any(k.startswith(p) for p in drop):
            n_dropped += 1
            continue
        new_k = remap_key(k, reverse=args.reverse)
        if new_k != k:
            n_remapped += 1
        out[new_k] = v.contiguous().clone()

    print(f"  remapped: {n_remapped} | dropped: {n_dropped} | output keys: {len(out)}", flush=True)

    if args.inplace:
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        save_file(out, str(tmp))
        tmp.replace(args.output)
        print(f"  wrote (in-place) {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_file(out, str(args.output))
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
