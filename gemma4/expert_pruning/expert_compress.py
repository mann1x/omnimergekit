#!/usr/bin/env python3
"""
Wanda structured neuron pruning for Gemma 4 MoE experts.

Prunes intermediate neurons globally per layer (same for all experts) using
contribution-weighted Wanda importance:
  importance[j] = sum_experts(contrib_w[e] * ||W[e,:,j]||_2 * ||X[e,j]||_2)

All layers keep the same count of neurons (HF config constraint: single
moe_intermediate_size). The contribution weighting ensures neurons important
for high-contribution experts are prioritized.

Usage:
  python expert_compress.py --model-path /path/to/109e --prune-fraction 0.1
  python expert_compress.py --model-path /path/to/109e --prune-fraction 0.2 \
      --calibration-file /tmp/cal_109e.json
"""

import argparse
import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

os.environ["HF_TOKEN"] = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
os.environ["CUDA_VISIBLE_DEVICES"] = ""

CALIBRATION_PROMPTS = [
    "Solve step by step: If f(x) = 3x^2 - 2x + 1, find f'(x) and evaluate at x=4.",
    "A bag contains 5 red and 3 blue balls. Draw 2 without replacement. P(both red)?",
    "Solve the differential equation dy/dx = y*sin(x), y(0) = 1.",
    "Find all prime numbers p such that p^2 + 2 is also prime.",
    "You have 8 identical coins, one heavier. Find it in 2 weighings.",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100?",
    "In a room of 23 people, what's the probability at least two share a birthday?",
    "Write a Python function to detect a cycle in a linked list using Floyd's algorithm.",
    "Implement binary search in Python. Handle edge cases.",
    "Write a function to find the longest common subsequence of two strings.",
    "Explain quantum entanglement in simple terms.",
    "How does CRISPR-Cas9 gene editing work?",
    "Describe the process of nuclear fusion in stars.",
    "Write a haiku about the ocean.",
    "Tell me a short story about a robot that learns to paint.",
    "Describe a sunset to someone who has never seen one.",
]


def parse_args():
    p = argparse.ArgumentParser(description="Wanda neuron pruning for MoE experts")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--prune-fraction", type=float, default=None,
                   help="Fraction of neurons to prune (e.g. 0.2 = 20%%)")
    p.add_argument("--target-intermediate", type=int, default=None,
                   help="Exact target intermediate size (must be divisible by 32)")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--cal-tokens", type=int, default=128)
    p.add_argument("--calibration-file", type=str, default=None)
    p.add_argument("--save-calibration", type=str, default=None)
    return p.parse_args()


def load_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model from {model_path} (fp16, CPU)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    print(f"Loaded in {time.time()-t0:.0f}s")
    return model, tokenizer


def calibrate(model, tokenizer, num_layers, num_experts, intermediate_size,
              max_new_tokens=128):
    """Two-phase calibration: generate then replay with hooks."""
    from torch import nn

    act_norms = [[torch.zeros(intermediate_size) for _ in range(num_experts)]
                 for _ in range(num_layers)]
    contrib_norms = [[0.0] * num_experts for _ in range(num_layers)]
    token_counts = [[0] * num_experts for _ in range(num_layers)]

    print(f"\nCalibrating with {len(CALIBRATION_PROMPTS)} prompts "
          f"({max_new_tokens} tokens each)...")

    for i, prompt in enumerate(CALIBRATION_PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
            add_generation_prompt=True, enable_thinking=True)
        input_ids = inputs["input_ids"]

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                 do_sample=False)
        gen_time = time.time() - t0
        gen_len = out.shape[1] - input_ids.shape[1]

        hooks = []
        for li in range(num_layers):
            layer = model.model.language_model.layers[li]
            if not hasattr(layer, "experts"):
                continue

            def make_hook(layer_idx):
                def hook(module, args, output):
                    hs, top_k_idx, top_k_wt = args
                    n = module.num_experts
                    with torch.no_grad():
                        mask = nn.functional.one_hot(
                            top_k_idx, num_classes=n).permute(2, 1, 0)
                        hit = torch.greater(
                            mask.sum(dim=(-1, -2)), 0).nonzero()
                        for eidx in hit:
                            eidx = eidx[0]
                            if eidx == n:
                                continue
                            eid = int(eidx)
                            pos, tidx = torch.where(mask[eidx])
                            cs = hs[tidx]
                            g, u = nn.functional.linear(
                                cs, module.gate_up_proj[eid]).chunk(2, dim=-1)
                            intermediate = module.act_fn(g) * u
                            act_norms[layer_idx][eid] += \
                                (intermediate ** 2).sum(dim=0).cpu()
                            token_counts[layer_idx][eid] += len(tidx)
                            expert_out = nn.functional.linear(
                                intermediate, module.down_proj[eid])
                            wt = top_k_wt[tidx, pos]
                            weighted = expert_out * wt.unsqueeze(-1)
                            contrib_norms[layer_idx][eid] += \
                                weighted.norm().item()
                return hook
            hooks.append(layer.experts.register_forward_hook(make_hook(li)))

        mm_ids = torch.zeros_like(out)
        t1 = time.time()
        with torch.no_grad():
            model(out, mm_token_type_ids=mm_ids)
        prof_time = time.time() - t1

        for h in hooks:
            h.remove()

        print(f"  [{i+1}/{len(CALIBRATION_PROMPTS)}] gen: {gen_len} tok {gen_time:.0f}s, "
              f"profile: {prof_time:.0f}s, seq: {out.shape[1]}")

    for li in range(num_layers):
        for eid in range(num_experts):
            act_norms[li][eid] = torch.sqrt(act_norms[li][eid])

    return act_norms, contrib_norms, token_counts


def save_calibration(filepath, act_norms, contrib_norms, token_counts,
                     num_layers, num_experts):
    data = {
        "num_layers": num_layers,
        "num_experts": num_experts,
        "act_norms": [[act_norms[li][eid].tolist() for eid in range(num_experts)]
                      for li in range(num_layers)],
        "contrib_norms": contrib_norms,
        "token_counts": token_counts,
    }
    with open(filepath, "w") as f:
        json.dump(data, f)
    print(f"Calibration saved to {filepath}")


def load_calibration(filepath):
    print(f"Loading calibration from {filepath}...")
    with open(filepath) as f:
        data = json.load(f)
    act_norms = [[torch.tensor(data["act_norms"][li][eid])
                  for eid in range(data["num_experts"])]
                 for li in range(data["num_layers"])]
    return act_norms, data["contrib_norms"], data["token_counts"]


def compute_wanda_masks(model_path, act_norms, contrib_norms,
                        num_layers, num_experts, intermediate_size,
                        target_intermediate):
    """
    Compute per-layer neuron keep masks using contribution-weighted Wanda.
    Same count of neurons kept per layer, but selection varies by importance.
    """
    source_dir = Path(model_path)
    with open(source_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_files = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_files[shard_name] = safe_open(
            str(source_dir / shard_name), framework="pt", device="cpu")

    prune_count = intermediate_size - target_intermediate
    keep_neurons = [None] * num_layers
    retained_importance = []

    print(f"\nComputing Wanda importance "
          f"(pruning {prune_count}/{intermediate_size} neurons per layer)...")

    for li in tqdm(range(num_layers), desc="Wanda masks"):
        key = f"model.language_model.layers.{li}.experts.down_proj"
        shard_name = weight_map[key]
        down_proj = shard_files[shard_name].get_tensor(key)

        layer_total = sum(contrib_norms[li])
        if layer_total > 0:
            cw = [c / layer_total for c in contrib_norms[li]]
        else:
            cw = [1.0 / num_experts] * num_experts

        importance = torch.zeros(intermediate_size)
        for eid in range(num_experts):
            w_norms = down_proj[eid].float().norm(dim=0)
            x_norms = act_norms[li][eid].float()
            score = cw[eid] * w_norms * x_norms
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            importance += score

        _, topk_idx = importance.topk(target_intermediate)
        keep = sorted(topk_idx.tolist())
        keep_neurons[li] = keep

        total_imp = importance.sum().item()
        kept_imp = importance[keep].sum().item()
        pct = kept_imp / max(total_imp, 1e-10) * 100
        retained_importance.append(pct)

    for sf in shard_files.values():
        del sf

    print(f"\n  Per-layer retained importance:")
    for li in range(num_layers):
        ri = retained_importance[li] if not np.isnan(retained_importance[li]) else 0.0
        bar = "█" * int(ri / 100 * 30)
        print(f"    L{li:2d}: {retained_importance[li]:6.2f}% {bar}")
    print(f"  Worst: L{np.argmin(retained_importance)} "
          f"at {min(retained_importance):.2f}%")
    print(f"  Mean: {np.mean(retained_importance):.2f}%")

    return keep_neurons


def apply_wanda(source_dir, output_dir, keep_neurons, num_layers,
                intermediate_size, target_intermediate):
    """Apply Wanda pruning — slice neurons from expert tensors."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(source_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    with open(source_dir / "config.json") as f:
        config = json.load(f)

    shard_files = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_files[shard_name] = safe_open(
            str(source_dir / shard_name), framework="pt", device="cpu")

    shard_keys = defaultdict(list)
    for key, shard in weight_map.items():
        shard_keys[shard].append(key)

    new_weight_map = {}
    current_shard = {}
    current_size = 0
    shard_idx = 1
    max_shard_bytes = int(5 * 1024**3)
    total_size = 0
    n_modified = 0

    print(f"\nApplying Wanda ({intermediate_size} -> {target_intermediate})...")

    for shard_name in tqdm(sorted(shard_keys.keys()), desc="Shards"):
        sf = shard_files[shard_name]
        for key in shard_keys[shard_name]:
            tensor = sf.get_tensor(key)

            m = re.match(
                r"model\.language_model\.layers\.(\d+)\.experts\.gate_up_proj",
                key)
            if m:
                li = int(m.group(1))
                keep = keep_neurons[li]
                gate_idx = keep
                up_idx = [k + intermediate_size for k in keep]
                tensor = tensor[:, gate_idx + up_idx, :]
                n_modified += 1

            m = re.match(
                r"model\.language_model\.layers\.(\d+)\.experts\.down_proj",
                key)
            if m:
                li = int(m.group(1))
                tensor = tensor[:, :, keep_neurons[li]]
                n_modified += 1

            tensor_size = tensor.numel() * tensor.element_size()
            if current_size + tensor_size > max_shard_bytes and current_shard:
                sf_name = f"model-{shard_idx:05d}.safetensors"
                save_file(current_shard, str(output_dir / sf_name))
                for k in current_shard:
                    new_weight_map[k] = sf_name
                shard_idx += 1
                current_shard = {}
                current_size = 0

            current_shard[key] = tensor
            current_size += tensor_size
            total_size += tensor_size

    if current_shard:
        sf_name = f"model-{shard_idx:05d}.safetensors"
        save_file(current_shard, str(output_dir / sf_name))
        for k in current_shard:
            new_weight_map[k] = sf_name

    for old_idx in range(1, shard_idx + 1):
        old_name = output_dir / f"model-{old_idx:05d}.safetensors"
        new_name = output_dir / f"model-{old_idx:05d}-of-{shard_idx:05d}.safetensors"
        old_name.rename(new_name)
        for k, v in new_weight_map.items():
            if v == f"model-{old_idx:05d}.safetensors":
                new_weight_map[k] = new_name.name

    for sf in shard_files.values():
        del sf

    new_index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    new_config = json.loads(json.dumps(config))
    new_config["text_config"]["moe_intermediate_size"] = target_intermediate
    with open(output_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    for fn in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "generation_config.json", "processor_config.json",
               "special_tokens_map.json", "preprocessor_config.json"]:
        src = source_dir / fn
        if src.exists():
            shutil.copy2(src, output_dir / fn)

    meta = {
        "base_model": str(source_dir),
        "method": "wanda_neuron_pruning",
        "original_intermediate": intermediate_size,
        "target_intermediate": target_intermediate,
        "prune_fraction": round(1 - target_intermediate / intermediate_size, 4),
        "per_layer_keep_neurons": {str(li): keep_neurons[li]
                                   for li in range(num_layers)},
        "total_params_bf16": total_size / 2,
        "total_size_gb": total_size / (1024 ** 3),
    }
    with open(output_dir / "compression_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Size: {total_size / 1024**3:.1f} GB "
          f"({total_size / 2 / 1e9:.1f}B params @ bf16)")
    print(f"  Shards: {shard_idx}")
    print(f"  Tensors modified: {n_modified}")
    print(f"{'='*60}")


def main():
    args = parse_args()
    source_dir = Path(args.model_path).resolve()

    with open(source_dir / "config.json") as f:
        config = json.load(f)
    text_cfg = config["text_config"]
    num_layers = text_cfg["num_hidden_layers"]
    num_experts = text_cfg["num_experts"]
    intermediate_size = text_cfg["moe_intermediate_size"]

    if args.target_intermediate:
        target_intermediate = args.target_intermediate
    elif args.prune_fraction:
        target_intermediate = int(intermediate_size * (1.0 - args.prune_fraction))
        # Round to nearest multiple of 32
        target_intermediate = (target_intermediate // 32) * 32
    else:
        raise ValueError("Must specify --target-intermediate or --prune-fraction")

    if target_intermediate % 32 != 0:
        raise ValueError(f"Target {target_intermediate} not divisible by 32. "
                         f"Nearest: {(target_intermediate // 32) * 32}")

    prune_frac = 1.0 - target_intermediate / intermediate_size
    print(f"Model: {source_dir}")
    print(f"  Layers: {num_layers}, Experts: {num_experts}, "
          f"Intermediate: {intermediate_size}")
    print(f"  Target: {target_intermediate} (prune {prune_frac:.1%})")

    # Calibration
    if args.calibration_file:
        act_norms, contrib_norms, token_counts = load_calibration(
            args.calibration_file)
    else:
        model, tokenizer = load_model(str(source_dir))
        act_norms, contrib_norms, token_counts = calibrate(
            model, tokenizer, num_layers, num_experts, intermediate_size,
            args.cal_tokens)
        del model, tokenizer
        import gc; gc.collect()

        if args.save_calibration:
            save_calibration(args.save_calibration, act_norms, contrib_norms,
                             token_counts, num_layers, num_experts)

    # Layer contribution summary
    layer_contribs = np.array([sum(contrib_norms[li]) for li in range(num_layers)])
    if layer_contribs.max() > 0:
        normalized = layer_contribs / layer_contribs.max()
    else:
        normalized = np.ones(num_layers)

    print(f"\n  Layer contribution (normalized):")
    for li in range(num_layers):
        bar = "█" * int(normalized[li] * 30)
        print(f"    L{li:2d}: {normalized[li]:.3f} {bar}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = source_dir.parent / f"gemma-4-A4B-109e-wanda-i{target_intermediate}"
    print(f"\n  Output: {output_dir}")

    # Size estimate
    orig_expert = num_experts * 3 * intermediate_size * 2816
    new_expert = num_experts * 3 * target_intermediate * 2816
    saved = (orig_expert - new_expert) * num_layers * 2  # bf16
    print(f"  Expert param savings: -{saved / 1024**3:.1f} GB bf16")

    # Compute masks and apply
    keep_neurons = compute_wanda_masks(
        str(source_dir), act_norms, contrib_norms,
        num_layers, num_experts, intermediate_size, target_intermediate)

    apply_wanda(
        str(source_dir), str(output_dir), keep_neurons,
        num_layers, intermediate_size, target_intermediate)


if __name__ == "__main__":
    main()
