#!/usr/bin/env python3
"""
Drop experts + build residual from top-solicited neurons of dropped experts.

Uses the improved expert+neuron analysis (scripts/expert_neuron_full.json) to:

1. Per-layer top-K expert selection (by aggregated wnorm across all categories)
2. From dropped experts in each layer, pick top intermediate_size neurons globally
   (by neuron_act activation magnitude)
3. Build a residual expert from those neurons (no clustering, direct copy)
4. Set residual router weights via calibration forward pass (TODO if requested)
   For now: residual router = average of dropped experts' router rows weighted by
   how many of each source's top neurons were kept

Usage:
  python expert_drop_residual.py --target-experts 96
  python expert_drop_residual.py --target-experts 88 --no-residual
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

os.environ["CUDA_VISIBLE_DEVICES"] = ""

DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"
DEFAULT_ANALYSIS_V3 = "scripts/expert_neuron_v4.json"
DEFAULT_ANALYSIS_OLD = "google/gemma-4-26B-A4B-it/eval_results/expert_contributions_full.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="Expert drop + residual from neuron analysis")
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--analysis-v3", type=str, default=DEFAULT_ANALYSIS_V3,
                   help="New analysis with per-neuron data")
    p.add_argument("--analysis-old", type=str, default=DEFAULT_ANALYSIS_OLD,
                   help="Old simple analysis (combine signals)")
    p.add_argument("--combine-weight", type=float, default=0.5,
                   help="Weight for v3 signal in combination (0=old only, 1=v3 only)")
    p.add_argument("--target-experts", type=int, required=True,
                   help="Number of experts to keep per layer (excludes residual)")
    p.add_argument("--no-residual", action="store_true",
                   help="Skip residual expert (pure expert drop)")
    p.add_argument("--residual-scale", type=float, default=0.1,
                   help="Scale factor for residual router weight (default 0.1)")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def load_v3(path, num_layers, num_experts):
    """Load v3 analysis (has metadata + per-neuron data)."""
    print(f"Loading v3 from {path}...")
    with open(path) as f:
        data = json.load(f)
    md = data["metadata"]
    intermediate_size = md["intermediate_size"]

    expert_wnorm = [[0.0] * num_experts for _ in range(num_layers)]
    neuron_act = [[None] * num_experts for _ in range(num_layers)]

    for cat, cat_data in data["categories"].items():
        for li_str, layer_experts in cat_data.items():
            li = int(li_str)
            for e in layer_experts:
                eid = e["id"]
                expert_wnorm[li][eid] += e["wnorm"]
                acts = torch.tensor(e["neuron_act"], dtype=torch.float32)
                if neuron_act[li][eid] is None:
                    neuron_act[li][eid] = acts
                else:
                    neuron_act[li][eid] += acts

    return expert_wnorm, neuron_act, intermediate_size


def load_old(path, num_layers, num_experts):
    """Load old simple analysis (no per-neuron data, no metadata)."""
    print(f"Loading old from {path}...")
    with open(path) as f:
        data = json.load(f)

    expert_wnorm = [[0.0] * num_experts for _ in range(num_layers)]
    for cat in data:
        for li_str in data[cat]:
            li = int(li_str)
            for e in data[cat][li_str]["experts"]:
                expert_wnorm[li][e["id"]] += e["wnorm"]
    return expert_wnorm


def normalize_wnorm(w):
    """Normalize per-layer so values sum to 1."""
    out = []
    for layer in w:
        s = sum(layer)
        if s > 0:
            out.append([x / s for x in layer])
        else:
            out.append(layer[:])
    return out


def combine_signals(v3, old, weight_v3, num_layers, num_experts):
    """Combine v3 and old wnorm signals via weighted average of normalized values."""
    v3_n = normalize_wnorm(v3)
    old_n = normalize_wnorm(old)
    combined = [[0.0] * num_experts for _ in range(num_layers)]
    for li in range(num_layers):
        for eid in range(num_experts):
            combined[li][eid] = (
                weight_v3 * v3_n[li][eid] + (1 - weight_v3) * old_n[li][eid])
    return combined


def select_experts(expert_wnorm, num_layers, num_experts, target_experts):
    """Per-layer top-K experts by wnorm."""
    keep_map = {}
    drop_map = {}
    for li in range(num_layers):
        sorted_e = sorted(range(num_experts),
                          key=lambda e: expert_wnorm[li][e], reverse=True)
        keep_map[li] = sorted(sorted_e[:target_experts])
        drop_map[li] = sorted(sorted_e[target_experts:])
    return keep_map, drop_map


def build_residual_neurons(neuron_act, drop_ids, intermediate_size):
    """
    From all dropped experts in a layer, pick top `intermediate_size` neurons globally
    by activation magnitude.

    Returns: list of (source_expert_id, neuron_idx_in_source) tuples,
             length = intermediate_size
    """
    candidates = []
    for eid in drop_ids:
        acts = neuron_act[eid]  # [intermediate_size]
        for n_idx in range(intermediate_size):
            candidates.append((float(acts[n_idx]), eid, n_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:intermediate_size]
    return [(eid, nidx) for _, eid, nidx in top]


def assemble_residual_expert(gate_up_proj, down_proj, residual_neurons,
                             intermediate_size, hidden_size):
    """
    Build new residual expert tensors from selected (source_expert, neuron) pairs.

    gate_up_proj: [num_experts, 2*intermediate, hidden]
    down_proj: [num_experts, hidden, intermediate]

    Returns: (gate_up_residual, down_residual)
      gate_up_residual: [2*intermediate, hidden]
      down_residual: [hidden, intermediate]
    """
    new_gate = torch.zeros(intermediate_size, hidden_size,
                           dtype=gate_up_proj.dtype)
    new_up = torch.zeros(intermediate_size, hidden_size,
                         dtype=gate_up_proj.dtype)
    new_down = torch.zeros(hidden_size, intermediate_size,
                           dtype=down_proj.dtype)

    for new_idx, (src_eid, src_nidx) in enumerate(residual_neurons):
        # gate_up_proj[src_eid] has shape [2*intermediate, hidden]
        # gate row src_nidx: gate_up_proj[src_eid, src_nidx, :]
        # up row src_nidx: gate_up_proj[src_eid, intermediate + src_nidx, :]
        new_gate[new_idx] = gate_up_proj[src_eid, src_nidx, :]
        new_up[new_idx] = gate_up_proj[src_eid, intermediate_size + src_nidx, :]
        # down_proj[src_eid, :, src_nidx] gives [hidden]
        new_down[:, new_idx] = down_proj[src_eid, :, src_nidx]

    gate_up_residual = torch.cat([new_gate, new_up], dim=0)
    return gate_up_residual, new_down


def main():
    args = parse_args()
    source_dir = Path(args.model_path).resolve()

    with open(source_dir / "config.json") as f:
        config = json.load(f)
    text_cfg = config["text_config"]
    num_layers = text_cfg["num_hidden_layers"]
    num_experts = text_cfg["num_experts"]
    intermediate_size = text_cfg["moe_intermediate_size"]
    hidden_size = text_cfg["hidden_size"]
    target_experts = args.target_experts
    has_residual = not args.no_residual
    actual_experts = target_experts + (1 if has_residual else 0)

    print(f"Model: {source_dir}")
    print(f"  Layers: {num_layers}, Experts: {num_experts}, "
          f"Intermediate: {intermediate_size}, Hidden: {hidden_size}")
    print(f"  Target: {target_experts} kept "
          f"{'+ 1 residual = ' + str(actual_experts) if has_residual else '(no residual)'}")

    # Load analyses
    v3_wnorm, neuron_act, ana_intermediate = load_v3(
        args.analysis_v3, num_layers, num_experts)
    assert ana_intermediate == intermediate_size
    old_wnorm = load_old(args.analysis_old, num_layers, num_experts)

    # Combine signals for expert selection
    combined_wnorm = combine_signals(
        v3_wnorm, old_wnorm, args.combine_weight, num_layers, num_experts)
    print(f"Combined signal: weight_v3={args.combine_weight}, "
          f"weight_old={1-args.combine_weight}")

    # Select experts using combined signal
    keep_map, drop_map = select_experts(
        combined_wnorm, num_layers, num_experts, target_experts)
    expert_wnorm = combined_wnorm  # for downstream code

    # Print per-layer summary
    print(f"\nPer-layer selection:")
    for li in range(num_layers):
        total = sum(expert_wnorm[li])
        kept = sum(expert_wnorm[li][e] for e in keep_map[li])
        pct = kept / max(total, 1e-10) * 100
        print(f"  L{li:2d}: keep {len(keep_map[li])}, "
              f"drop {len(drop_map[li])}, {pct:.2f}% wnorm retained")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "dr" if has_residual else "drop"
        output_dir = source_dir.parent / f"gemma-4-A4B-{target_experts}e-{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")

    # Load safetensors
    with open(source_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_files = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_files[shard_name] = safe_open(
            str(source_dir / shard_name), framework="pt", device="cpu")

    shard_keys = defaultdict(list)
    for key, shard in weight_map.items():
        shard_keys[shard].append(key)

    # Pre-compute residual expert per layer
    residual_data = {}  # li -> (gate_up_residual, down_residual, source_counts)
    if has_residual:
        print(f"\nBuilding residual experts...")
        for li in tqdm(range(num_layers), desc="Residual"):
            gu_key = f"model.language_model.layers.{li}.experts.gate_up_proj"
            dn_key = f"model.language_model.layers.{li}.experts.down_proj"
            gate_up = shard_files[weight_map[gu_key]].get_tensor(gu_key)
            down = shard_files[weight_map[dn_key]].get_tensor(dn_key)

            # Pick top neurons from dropped experts
            top_neurons = build_residual_neurons(
                neuron_act[li], drop_map[li], intermediate_size)

            # Count how many neurons came from each source expert
            source_counts = defaultdict(int)
            for src_eid, _ in top_neurons:
                source_counts[src_eid] += 1

            # Assemble residual expert tensors
            gate_up_res, down_res = assemble_residual_expert(
                gate_up, down, top_neurons, intermediate_size, hidden_size)

            residual_data[li] = (gate_up_res, down_res, dict(source_counts))

    # Process and write tensors
    new_weight_map = {}
    current_shard = {}
    current_size = 0
    shard_idx = 1
    max_shard_bytes = int(5 * 1024**3)
    total_size = 0

    print(f"\nWriting tensors...")
    for shard_name in tqdm(sorted(shard_keys.keys()), desc="Shards"):
        sf = shard_files[shard_name]
        for key in shard_keys[shard_name]:
            tensor = sf.get_tensor(key)

            # Expert gate_up_proj: slice keep + append residual
            m = re.match(
                r"model\.language_model\.layers\.(\d+)\.experts\.gate_up_proj",
                key)
            if m:
                li = int(m.group(1))
                kept = tensor[sorted(keep_map[li])]  # [target_experts, 2*int, hidden]
                if has_residual:
                    res_gu = residual_data[li][0].to(tensor.dtype)
                    tensor = torch.cat([kept, res_gu.unsqueeze(0)], dim=0)
                else:
                    tensor = kept

            # Expert down_proj
            m = re.match(
                r"model\.language_model\.layers\.(\d+)\.experts\.down_proj",
                key)
            if m:
                li = int(m.group(1))
                kept = tensor[sorted(keep_map[li])]  # [target_experts, hidden, int]
                if has_residual:
                    res_dn = residual_data[li][1].to(tensor.dtype)
                    tensor = torch.cat([kept, res_dn.unsqueeze(0)], dim=0)
                else:
                    tensor = kept

            # Router proj.weight: keep + residual row
            m = re.match(
                r"model\.language_model\.layers\.(\d+)\.router\.proj\.weight",
                key)
            if m:
                li = int(m.group(1))
                orig = tensor.float()
                kept_rows = orig[sorted(keep_map[li])]
                if has_residual:
                    # Residual router = weighted avg of source experts, scaled down
                    src_counts = residual_data[li][2]
                    total = sum(src_counts.values())
                    residual_row = torch.zeros_like(orig[0])
                    for src, cnt in src_counts.items():
                        residual_row += orig[src] * (cnt / total)
                    # Scale down so router rarely picks the residual expert
                    residual_row = residual_row * args.residual_scale
                    residual_row = residual_row.unsqueeze(0)
                    tensor = torch.cat([kept_rows, residual_row], dim=0).to(
                        torch.bfloat16)
                else:
                    tensor = kept_rows.to(torch.bfloat16)

            # Router per_expert_scale
            m = re.match(
                r"model\.language_model\.layers\.(\d+)\.router\.per_expert_scale",
                key)
            if m:
                li = int(m.group(1))
                kept = tensor[sorted(keep_map[li])]
                if has_residual:
                    src_counts = residual_data[li][2]
                    total = sum(src_counts.values())
                    residual_pes = sum(
                        tensor[src].float() * (cnt / total)
                        for src, cnt in src_counts.items())
                    # Also scale per_expert_scale by the same factor
                    residual_pes = residual_pes * args.residual_scale
                    residual_pes = residual_pes.unsqueeze(0).to(tensor.dtype)
                    tensor = torch.cat([kept, residual_pes])
                else:
                    tensor = kept

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
    new_config["text_config"]["num_experts"] = actual_experts
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
        "method": "expert_drop_with_neuron_residual",
        "analysis_v3": args.analysis_v3,
        "analysis_old": args.analysis_old,
        "combine_weight": args.combine_weight,
        "original_experts": num_experts,
        "target_experts": target_experts,
        "actual_experts": actual_experts,
        "has_residual": has_residual,
        "intermediate_size": intermediate_size,
        "per_layer_keep": {str(li): keep_map[li] for li in range(num_layers)},
        "per_layer_drop": {str(li): drop_map[li] for li in range(num_layers)},
        "total_size_gb": total_size / (1024 ** 3),
    }
    if has_residual:
        meta["residual_source_counts"] = {
            str(li): residual_data[li][2] for li in range(num_layers)
        }
    with open(output_dir / "drop_residual_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Size: {total_size / 1024**3:.1f} GB "
          f"({total_size / 2 / 1e9:.1f}B params @ bf16)")
    print(f"  Experts: {actual_experts} "
          f"({target_experts} kept{' + 1 residual' if has_residual else ''})")
    print(f"  Shards: {shard_idx}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
