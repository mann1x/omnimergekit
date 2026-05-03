#!/usr/bin/env python3
"""
Pre-compute diagonal Fisher information for a model on calibration data.

Uses full-model forward+backward on CPU with gradient checkpointing.
Each sample's gradients are accumulated in fp32. GPU is NOT used — this
runs entirely on CPU to avoid OOM issues with backward pass.

For a 27B model with 64 samples × 256 tokens: ~30-60 min on a modern CPU.
"""
import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch


def load_calibration_texts(cal_path: str, num_samples: int) -> List[str]:
    texts = []
    with open(cal_path) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
            if len(texts) >= num_samples:
                break
    return texts


def main():
    ap = argparse.ArgumentParser(description="Pre-compute diagonal Fisher information")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--cal-data", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default=None,
                    help="cpu | cuda. Default: auto (cuda if available, else cpu). "
                         "27B+ models almost always need cpu (FP32 grads don't fit GPU). "
                         "4B-13B with fp32 grads fits 48GB GPU comfortably and runs ~10x faster.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="Compute dtype. fp32 is the safe default (Fisher needs accurate grads). "
                         "bf16 trades noise for memory — only use if fp32 OOMs on GPU.")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"=== Fisher pre-computation ({args.device}, {args.dtype}) ===")
    print(f"  model      : {args.model}")
    print(f"  num-samples: {args.num_samples}")
    print(f"  max-length : {args.max_length}")
    print(flush=True)

    texts = load_calibration_texts(str(args.cal_data), args.num_samples)
    print(f"  {len(texts)} calibration texts", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-tokenize
    cal_tokens = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=args.max_length, padding=False)["input_ids"]
        if ids.shape[1] >= 2:
            cal_tokens.append(ids)
    print(f"  {len(cal_tokens)} valid samples", flush=True)

    # Load model on the chosen device. For multimodal models like Qwen3_5
    # that lack an AutoModelForCausalLM mapping, fall back to the explicit
    # ConditionalGeneration class. Forward with text-only input still gives
    # us LM-head loss; vision tower stays unused (no image input → no grads).
    print(f"Loading model on {args.device} ({args.dtype})...", flush=True)
    cfg = AutoConfig.from_pretrained(str(args.model))
    arch = cfg.architectures[0] if cfg.architectures else ""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model),
            torch_dtype=compute_dtype,
            device_map=args.device,
            attn_implementation="eager",
        )
    except (KeyError, ValueError, ModuleNotFoundError) as e:
        print(f"  AutoModelForCausalLM unavailable ({e}); falling back to {arch}", flush=True)
        import importlib
        # Resolve class by name from transformers; works for Qwen3_5ForConditionalGeneration etc.
        try:
            cls = getattr(importlib.import_module("transformers"), arch)
        except AttributeError:
            raise RuntimeError(
                f"Cannot resolve {arch} in transformers — install a version that ships it,"
                " or use trust_remote_code with the model's own modeling.py")
        model = cls.from_pretrained(
            str(args.model),
            torch_dtype=compute_dtype,
            device_map=args.device,
            attn_implementation="eager",
        )

    # Enable gradient checkpointing to reduce memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled", flush=True)

    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    param_names = [n for n, p in model.named_parameters() if p.is_floating_point()]
    print(f"  {len(param_names)} trainable parameters", flush=True)

    fisher: Dict[str, torch.Tensor] = {}
    n_processed = 0
    t_start = time.time()

    print(f"Computing Fisher ({len(cal_tokens)} samples, {args.device})...", flush=True)
    for i, input_ids in enumerate(cal_tokens):
        try:
            input_ids = input_ids.to(args.device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Always accumulate Fisher in fp32 on CPU — keeps numerical
                    # precision regardless of compute device, and frees GPU.
                    grad_sq = param.grad.detach().to(torch.float32).pow(2).cpu()
                    if name not in fisher:
                        fisher[name] = grad_sq.clone()
                    else:
                        fisher[name] += grad_sq

            model.zero_grad(set_to_none=True)
            n_processed += 1

        except Exception as e:
            print(f"  Sample {i} failed: {e}", flush=True)
            model.zero_grad(set_to_none=True)
            continue

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(cal_tokens) - i - 1) / rate
            print(f"  [{i+1}/{len(cal_tokens)}] {n_processed} ok, {elapsed:.0f}s elapsed, {eta:.0f}s remaining", flush=True)

    print(f"\nProcessed {n_processed}/{len(cal_tokens)} samples", flush=True)
    if n_processed == 0:
        print("ERROR: no samples processed", file=sys.stderr)
        sys.exit(1)

    for name in fisher:
        fisher[name] /= n_processed

    print(f"  Fisher computed for {len(fisher)} parameters", flush=True)

    from safetensors.torch import save_file
    save_file(fisher, str(args.output))
    size_mb = args.output.stat().st_size / 1024**2
    elapsed = time.time() - t_start
    print(f"\n=== DONE in {elapsed:.0f}s ({elapsed/60:.1f} min) ===")
    print(f"  Output: {args.output} ({size_mb:.1f} MB)")
    print(f"  Parameters: {len(fisher)}")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
