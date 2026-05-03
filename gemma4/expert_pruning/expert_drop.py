#!/usr/bin/env python3
"""
Drop least-contributing experts from Gemma 4 26B-A4B based on contribution analysis.
Uses per-layer drop maps to remove experts and remap router weights.

For each layer:
  - gate_up_proj: [128, 1408, 2816] → [N, 1408, 2816]  (keep only N)
  - down_proj: [128, 2816, 704] → [N, 2816, 704]
  - router.proj.weight: [128, 2816] → [N, 2816]
  - router.per_expert_scale: [128] → [N]

Usage:
  python expert_drop.py                                      # legacy default: 109e from eval_results/expert_drop_map_109.json (cwd=128e dir)
  python expert_drop.py --drop-map scripts/hybrid_120e_drop_map.json --suffix -hybrid
  python expert_drop.py --source-dir google/gemma-4-26B-A4B-it --drop-map scripts/hybrid_120e_drop_map.json --suffix -hybrid
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=str, default=".",
                    help="path to base model dir (default: cwd)")
    ap.add_argument("--drop-map", type=str, default=None,
                    help="path to drop map JSON (default: <source>/eval_results/expert_drop_map_109.json)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="output dir (default: <source>.parent/gemma-4-A4B-{N}e{suffix})")
    ap.add_argument("--suffix", type=str, default="",
                    help="suffix for default output name, e.g. -hybrid")
    return ap.parse_args()


def main():
    args = parse_args()

    source_dir = Path(args.source_dir).resolve()
    drop_map_file = Path(args.drop_map) if args.drop_map else (source_dir / "eval_results" / "expert_drop_map_109.json")
    drop_map_file = drop_map_file.resolve()

    with open(drop_map_file) as f:
        drop_map = {int(k): v for k, v in json.load(f).items()}

    with open(source_dir / "config.json") as f:
        config = json.load(f)

    text_cfg = config["text_config"]
    num_layers = text_cfg["num_hidden_layers"]
    num_experts_orig = text_cfg["num_experts"]
    target_experts = num_experts_orig - len(drop_map[0])  # all layers drop same count

    print(f"Source:    {source_dir}")
    print(f"Drop map:  {drop_map_file}")
    print(f"Experts:   {num_experts_orig} → {target_experts}")
    print(f"Layers:    {num_layers}")

    # Build per-layer keep indices (sorted for consistent ordering)
    keep_map = {}
    for li in range(num_layers):
        drop_set = set(drop_map[li])
        keep_map[li] = sorted(set(range(num_experts_orig)) - drop_set)
        assert len(keep_map[li]) == target_experts, f"Layer {li}: expected {target_experts} keep, got {len(keep_map[li])}"

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = source_dir.parent / f"gemma-4-A4B-{target_experts}e{args.suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output:    {output_dir}")

    # Load index
    with open(source_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Open all shard files
    shard_files = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_files[shard_name] = safe_open(str(source_dir / shard_name), framework="pt", device="cpu")

    # Group keys by shard
    shard_keys = defaultdict(list)
    for key, shard in weight_map.items():
        shard_keys[shard].append(key)

    # Process
    new_weight_map = {}
    current_shard = {}
    current_size = 0
    shard_idx = 1
    max_shard_bytes = int(5 * 1024**3)  # 5GB shards
    total_size = 0
    n_expert_tensors = 0
    n_router_tensors = 0

    for shard_name in tqdm(sorted(shard_keys.keys()), desc="Processing"):
        sf = shard_files[shard_name]
        for key in shard_keys[shard_name]:
            tensor = sf.get_tensor(key)

            # Expert stacked weights: gate_up_proj or down_proj
            m_expert = re.match(
                r"model\.language_model\.layers\.(\d+)\.experts\.(gate_up_proj|down_proj)",
                key
            )
            if m_expert:
                layer_idx = int(m_expert.group(1))
                keep_ids = keep_map[layer_idx]
                tensor = tensor[keep_ids]  # [128,...] → [109,...]
                n_expert_tensors += 1

            # Router proj.weight
            m_router_proj = re.match(
                r"model\.language_model\.layers\.(\d+)\.router\.proj\.weight",
                key
            )
            if m_router_proj:
                layer_idx = int(m_router_proj.group(1))
                keep_ids = keep_map[layer_idx]
                tensor = tensor[keep_ids]  # [128, hidden] → [109, hidden]
                n_router_tensors += 1

            # Router per_expert_scale
            m_router_scale = re.match(
                r"model\.language_model\.layers\.(\d+)\.router\.per_expert_scale",
                key
            )
            if m_router_scale:
                layer_idx = int(m_router_scale.group(1))
                keep_ids = keep_map[layer_idx]
                tensor = tensor[keep_ids]  # [128] → [109]
                n_router_tensors += 1

            # Write to shard
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

    # Final shard
    if current_shard:
        sf_name = f"model-{shard_idx:05d}.safetensors"
        save_file(current_shard, str(output_dir / sf_name))
        for k in current_shard:
            new_weight_map[k] = sf_name

    # Rename shards with total count
    for old_idx in range(1, shard_idx + 1):
        old_name = output_dir / f"model-{old_idx:05d}.safetensors"
        new_name = output_dir / f"model-{old_idx:05d}-of-{shard_idx:05d}.safetensors"
        old_name.rename(new_name)
        for k, v in new_weight_map.items():
            if v == f"model-{old_idx:05d}.safetensors":
                new_weight_map[k] = new_name.name

    # Close source files
    for sf in shard_files.values():
        del sf

    # Write index
    new_index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # Update config
    new_config = json.loads(json.dumps(config))
    new_config["text_config"]["num_experts"] = target_experts
    with open(output_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    # Copy non-weight files
    for fn in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "generation_config.json", "processor_config.json",
               "special_tokens_map.json", "preprocessor_config.json"]:
        src = source_dir / fn
        if src.exists():
            shutil.copy2(src, output_dir / fn)

    # Save metadata
    meta = {
        "base_model": "google/gemma-4-26B-A4B-it",
        "method": "expert_drop_by_contribution",
        "drop_map_file": str(drop_map_file),
        "original_experts": num_experts_orig,
        "target_experts": target_experts,
        "per_layer_keep": {str(li): ids for li, ids in keep_map.items()},
        "per_layer_drop": {str(li): ids for li, ids in drop_map.items()},
    }
    with open(output_dir / "expert_drop_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone!")
    print(f"  Expert tensors pruned: {n_expert_tensors}")
    print(f"  Router tensors pruned: {n_router_tensors}")
    print(f"  Total size: {total_size / 1024**3:.1f} GB ({total_size / 2 / 1e9:.1f}B params @ bf16)")
    print(f"  Shards: {shard_idx}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
