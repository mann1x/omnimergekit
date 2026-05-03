#!/usr/bin/env python3
"""
DERN v2 — bugfixed implementation per arXiv:2509.10377.

Differences from v1:
1. NO residual expert (pure neuron redistribution into surviving experts)
2. Threshold 0.0 (assign every dropped segment to best-match surviving expert)
3. Router update is proportional per transfer count: G_dst += (n/intermediate) * G_src
4. K-means with norm equalization (per DERN Eq. 11) — already in v1
5. Uses v4 analysis (the proven OLD methodology with per-neuron data)

Algorithm:
- Drop 32 experts per layer based on v4 wnorm picks
- For each dropped expert, decompose into 704 neuron segments
  segment_i = (gate_row_i, up_row_i, down_col_i)
- For each segment, find the surviving expert with max cosine similarity
- Each surviving expert collects: its original 704 + all transferred segments
- Spherical weighted k-means with norm equalization → compress back to 704
- Router: G_{E_r} += (n_transferred_from_E_o / 704) * G_{E_o}

Usage:
  python expert_dern_v2.py --target-experts 96 \\
    --analysis scripts/expert_neuron_v4.json
"""

import argparse
import json
import os
import re
import shutil
import time
from collections import defaultdict, Counter
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def parse_args():
    p = argparse.ArgumentParser(description="DERN v2 bugfixed")
    p.add_argument("--model-path", type=str,
                   default="google/gemma-4-26B-A4B-it")
    p.add_argument("--analysis", type=str,
                   default="scripts/expert_neuron_v4.json")
    p.add_argument("--target-experts", type=int, required=True)
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Min cosine sim to transfer (0.0 = transfer everything)")
    p.add_argument("--kmeans-iters", type=int, default=30)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def load_analysis(path, num_layers, num_experts):
    """Load wnorm aggregated across categories."""
    print(f"Loading analysis from {path}...")
    with open(path) as f:
        data = json.load(f)
    md = data.get("metadata", {})
    intermediate_size = md.get("intermediate_size", 704)

    expert_wnorm = [[0.0] * num_experts for _ in range(num_layers)]
    for cat in data["categories"]:
        for li_str, experts in data["categories"][cat].items():
            for e in experts:
                expert_wnorm[int(li_str)][e["id"]] += e["wnorm"]
    return expert_wnorm, intermediate_size


def select_experts(wnorm, num_layers, num_experts, target):
    keep_map, drop_map = {}, {}
    for li in range(num_layers):
        sorted_e = sorted(range(num_experts),
                          key=lambda e: wnorm[li][e], reverse=True)
        keep_map[li] = sorted(sorted_e[:target])
        drop_map[li] = sorted(sorted_e[target:])
    return keep_map, drop_map


def extract_segments(gate_up_row, down_row, intermediate_size):
    """Return list of (gate_vec, up_vec, down_vec) per neuron."""
    gate = gate_up_row[:intermediate_size]
    up = gate_up_row[intermediate_size:]
    return [(gate[i], up[i], down_row[:, i]) for i in range(intermediate_size)]


def assign_to_survivors(dropped_segments, dropped_sources,
                        surv_segments_by_eid, keep_ids, threshold):
    """
    For each dropped segment, find max-cosine surviving expert.
    Returns: assignments {eid: list of (segment, source_eid)},
             transfer_counts {src: {dst: count}}, n_assigned
    """
    if not dropped_segments:
        return {eid: [] for eid in keep_ids}, defaultdict(lambda: defaultdict(int)), 0

    # Build dropped matrices [N, 3*hidden]
    d_vecs = torch.stack([torch.cat([s[0], s[1], s[2]]) for s in dropped_segments])
    d_norm = torch.nn.functional.normalize(d_vecs, dim=1)

    n_dropped = len(dropped_segments)
    best_sim = torch.full((n_dropped,), -2.0)
    best_eid = torch.full((n_dropped,), -1, dtype=torch.long)

    for eid in keep_ids:
        s_vecs = torch.stack([torch.cat([s[0], s[1], s[2]])
                              for s in surv_segments_by_eid[eid]])
        s_norm = torch.nn.functional.normalize(s_vecs, dim=1)
        # [N_dropped, N_surv]
        sims = d_norm @ s_norm.T
        max_per_dropped, _ = sims.max(dim=1)
        better = max_per_dropped > best_sim
        best_sim[better] = max_per_dropped[better]
        best_eid[better] = eid

    assignments = {eid: [] for eid in keep_ids}
    transfer_counts = defaultdict(lambda: defaultdict(int))
    n_assigned = 0
    for i in range(n_dropped):
        if best_sim[i] >= threshold and best_eid[i] >= 0:
            dst = int(best_eid[i])
            src = dropped_sources[i]
            assignments[dst].append((dropped_segments[i], src))
            transfer_counts[src][dst] += 1
            n_assigned += 1

    return assignments, transfer_counts, n_assigned


def spherical_kmeans(segments, weights, k, max_iter=30, tol=1e-4):
    """Weighted spherical k-means with norm equalization (DERN Eq. 11)."""
    hidden = segments[0][0].shape[0]
    n = len(segments)

    data = torch.stack([torch.cat([s[0], s[1], s[2]]) for s in segments])
    w = torch.tensor(weights, dtype=torch.float32)

    if n <= k:
        result = [(s[0].clone(), s[1].clone(), s[2].clone()) for s in segments]
        while len(result) < k:
            result.append((torch.zeros(hidden), torch.zeros(hidden), torch.zeros(hidden)))
        return result

    # Init centroids: top-k by activation magnitude
    magnitudes = torch.tensor([s[0].abs().max().item() + s[0].abs().min().abs().item()
                               for s in segments])
    _, init_idx = magnitudes.topk(k)
    centroids = data[init_idx].clone()

    for it in range(max_iter):
        data_norm = torch.nn.functional.normalize(data, dim=1)
        cent_norm = torch.nn.functional.normalize(centroids, dim=1)
        sims = data_norm @ cent_norm.T
        labels = sims.argmax(dim=1)

        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.sum() == 0:
                new_centroids[j] = centroids[j]
                continue

            cluster_data = data[mask]
            cluster_w = w[mask]

            # Norm equalization (DERN Eq. 11)
            norms = cluster_data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            mean_norm = norms.mean()
            equalized = cluster_data * (mean_norm / norms)

            cluster_w_norm = cluster_w / cluster_w.sum().clamp(min=1e-8)
            centroid = (equalized * cluster_w_norm.unsqueeze(1)).sum(dim=0)
            cn = centroid.norm().clamp(min=1e-8)
            new_centroids[j] = (centroid / cn) * mean_norm

        shift = (new_centroids - centroids).norm()
        centroids = new_centroids
        if shift < tol:
            break

    result = []
    for j in range(k):
        c = centroids[j]
        result.append((c[:hidden], c[hidden:2*hidden], c[2*hidden:]))
    return result


def process_layer(gate_up_proj, down_proj, keep_ids, drop_ids, expert_wnorm,
                  intermediate_size, hidden_size, threshold, kmeans_iters):
    """Process one layer: redistribute neurons, k-means survivors. Returns new tensors and transfer counts."""
    # Extract all segments
    all_segs = {}
    for eid in list(keep_ids) + list(drop_ids):
        gu = gate_up_proj[eid].float()
        dn = down_proj[eid].float()
        all_segs[eid] = extract_segments(gu, dn, intermediate_size)

    # Collect dropped segments
    dropped_segments = []
    dropped_sources = []
    for eid in drop_ids:
        for seg in all_segs[eid]:
            dropped_segments.append(seg)
            dropped_sources.append(eid)

    surv_segs = {eid: all_segs[eid] for eid in keep_ids}
    assignments, transfer_counts, n_assigned = assign_to_survivors(
        dropped_segments, dropped_sources, surv_segs, keep_ids, threshold)

    # K-means within each surviving expert
    new_gate_up_list = []
    new_down_list = []
    for eid in sorted(keep_ids):
        original = all_segs[eid]
        transferred = [t[0] for t in assignments[eid]]
        all_for_expert = original + transferred

        imp = expert_wnorm[eid]
        weights = [imp] * len(original)
        for _ in transferred:
            weights.append(imp * 0.5)  # transferred segments weighted lower

        if len(all_for_expert) > intermediate_size:
            centroids = spherical_kmeans(
                all_for_expert, weights, intermediate_size,
                max_iter=kmeans_iters)
        else:
            centroids = [(s[0].clone(), s[1].clone(), s[2].clone())
                         for s in all_for_expert]
            while len(centroids) < intermediate_size:
                centroids.append((torch.zeros(hidden_size),
                                  torch.zeros(hidden_size),
                                  torch.zeros(hidden_size)))

        gate_rows = torch.stack([c[0] for c in centroids])
        up_rows = torch.stack([c[1] for c in centroids])
        down_cols = torch.stack([c[2] for c in centroids])

        new_gate_up_list.append(torch.cat([gate_rows, up_rows], dim=0))
        new_down_list.append(down_cols.T)

    new_gate_up = torch.stack(new_gate_up_list)
    new_down = torch.stack(new_down_list)

    return new_gate_up, new_down, n_assigned, dict(transfer_counts)


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
    target = args.target_experts

    print(f"Model: {source_dir}")
    print(f"  Layers={num_layers}, Experts={num_experts}, Intermediate={intermediate_size}")
    print(f"  Target experts: {target}")
    print(f"  Similarity threshold: {args.threshold}")

    # Load analysis
    expert_wnorm, _ = load_analysis(args.analysis, num_layers, num_experts)
    keep_map, drop_map = select_experts(
        expert_wnorm, num_layers, num_experts, target)

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = source_dir.parent / f"gemma-4-A4B-{target}e-dern-v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}\n")

    # Open shards
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

    # Pass 1: Process each layer
    print(f"Pass 1: DERN processing {num_layers} layers...")
    gate_up_cache = {}
    down_cache = {}
    transfer_per_layer = {}
    total_assigned = 0

    for li in tqdm(range(num_layers), desc="DERN"):
        gu_key = f"model.language_model.layers.{li}.experts.gate_up_proj"
        dn_key = f"model.language_model.layers.{li}.experts.down_proj"
        gate_up = shard_files[weight_map[gu_key]].get_tensor(gu_key)
        down = shard_files[weight_map[dn_key]].get_tensor(dn_key)

        t0 = time.time()
        new_gu, new_dn, n_asgn, transfer_counts = process_layer(
            gate_up, down,
            keep_map[li], drop_map[li],
            expert_wnorm[li],
            intermediate_size, hidden_size,
            args.threshold, args.kmeans_iters)
        elapsed = time.time() - t0

        gate_up_cache[li] = new_gu.to(torch.bfloat16)
        down_cache[li] = new_dn.to(torch.bfloat16)
        transfer_per_layer[li] = transfer_counts
        total_assigned += n_asgn
        tqdm.write(f"  L{li:2d}: {n_asgn} assigned, {elapsed:.0f}s")

    # Pass 2: Write tensors
    print(f"\nPass 2: Writing tensors...")
    new_weight_map = {}
    current_shard = {}
    current_size = 0
    shard_idx = 1
    max_shard_bytes = int(5 * 1024**3)
    total_size = 0

    for shard_name in tqdm(sorted(shard_keys.keys()), desc="Shards"):
        sf = shard_files[shard_name]
        for key in shard_keys[shard_name]:
            tensor = sf.get_tensor(key)

            m = re.match(r"model\.language_model\.layers\.(\d+)\.experts\.gate_up_proj", key)
            if m:
                tensor = gate_up_cache.pop(int(m.group(1)))

            m = re.match(r"model\.language_model\.layers\.(\d+)\.experts\.down_proj", key)
            if m:
                tensor = down_cache.pop(int(m.group(1)))

            # Router proj.weight: proportional update per DERN formula
            m = re.match(r"model\.language_model\.layers\.(\d+)\.router\.proj\.weight", key)
            if m:
                li = int(m.group(1))
                orig = tensor.float()
                updated = orig.clone()
                # G_{E_r} += (n_transferred / intermediate_size) * G_{E_o}
                for src_eid, dst_counts in transfer_per_layer[li].items():
                    src_row = orig[src_eid]
                    for dst_eid, n in dst_counts.items():
                        updated[dst_eid] += src_row * (n / intermediate_size)
                tensor = updated[sorted(keep_map[li])].to(torch.bfloat16)

            # Router per_expert_scale: just slice (no transfer formula in paper)
            m = re.match(r"model\.language_model\.layers\.(\d+)\.router\.per_expert_scale", key)
            if m:
                li = int(m.group(1))
                tensor = tensor[sorted(keep_map[li])]

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
    new_config["text_config"]["num_experts"] = target
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
        "method": "DERN_v2",
        "analysis": args.analysis,
        "original_experts": num_experts,
        "target_experts": target,
        "intermediate_size": intermediate_size,
        "threshold": args.threshold,
        "kmeans_iters": args.kmeans_iters,
        "total_segments_assigned": total_assigned,
        "per_layer_keep": {str(li): keep_map[li] for li in range(num_layers)},
        "per_layer_drop": {str(li): drop_map[li] for li in range(num_layers)},
        "total_size_gb": total_size / (1024 ** 3),
    }
    with open(output_dir / "dern_v2_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Size: {total_size / 1024**3:.1f} GB ({total_size / 2 / 1e9:.1f}B params)")
    print(f"  Shards: {shard_idx}")
    print(f"  Experts: {target}")
    print(f"  Total segments assigned: {total_assigned}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
