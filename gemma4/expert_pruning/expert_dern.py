#!/usr/bin/env python3
"""
DERN: Drop Experts, Recombine Neurons for Gemma 4 MoE.

Algorithm (arXiv:2509.10377):
1. Score experts by contribution, select top-K to keep per layer
2. Decompose dropped experts into neuron segments (gate_row, up_row, down_col)
3. Assign compatible segments to surviving experts (cosine similarity > threshold)
4. Cluster surviving expert's segments back to original intermediate_size (spherical k-means)
5. Update router weights proportionally
6. Save pruned model

Usage:
  python expert_dern.py --model-path google/gemma-4-26B-A4B-it --target-experts 96
  python expert_dern.py --model-path google/gemma-4-26B-A4B-it --target-experts 96 \
      --contribution-file eval_results/expert_contributions_full.json
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


def parse_args():
    p = argparse.ArgumentParser(description="DERN expert pruning for MoE")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--target-experts", type=int, required=True,
                   help="Number of experts to keep per layer")
    p.add_argument("--contribution-file", type=str, default=None,
                   help="Pre-computed expert contributions JSON")
    p.add_argument("--similarity-threshold", type=float, default=0.5,
                   help="Min cosine similarity to transfer a segment (default: 0.5)")
    p.add_argument("--cluster-ratio", type=float, default=1.0,
                   help="Output intermediate = original * ratio (default: 1.0)")
    p.add_argument("--kmeans-iters", type=int, default=30,
                   help="Max k-means iterations (default: 30)")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def load_contributions(contrib_file, num_layers, num_experts):
    """Load and aggregate contribution data across topics."""
    with open(contrib_file) as f:
        data = json.load(f)

    # Aggregate weighted output norms across all topics
    agg = [[0.0] * num_experts for _ in range(num_layers)]
    for topic in data:
        for li_str, layer_data in data[topic].items():
            li = int(li_str)
            for e in layer_data["experts"]:
                agg[li][e["id"]] += e["wnorm"]
    return agg


def select_experts(contributions, num_layers, num_experts, target_experts):
    """Select top-K experts per layer by contribution."""
    keep_map = {}
    drop_map = {}
    for li in range(num_layers):
        scored = sorted(range(num_experts), key=lambda e: contributions[li][e],
                        reverse=True)
        keep_map[li] = sorted(scored[:target_experts])
        drop_map[li] = sorted(scored[target_experts:])
    return keep_map, drop_map


def extract_segments(gate_up_row, down_row, intermediate_size):
    """
    Extract neuron segments from one expert.
    gate_up_row: [2*intermediate, hidden] (gate then up stacked)
    down_row: [hidden, intermediate]

    Returns: list of (gate_vec, up_vec, down_vec) tuples, each vec shape [hidden]
    """
    gate = gate_up_row[:intermediate_size]     # [intermediate, hidden]
    up = gate_up_row[intermediate_size:]        # [intermediate, hidden]
    # down_row[:, i] gives [hidden] for neuron i
    segments = []
    for i in range(intermediate_size):
        segments.append((gate[i], up[i], down_row[:, i]))
    return segments


def segments_to_matrix(segments):
    """Convert list of (gate, up, down) to stacked matrices for vectorized ops.
    Returns: up_matrix [N, hidden], down_matrix [N, hidden]
    (gate excluded from similarity per paper: gate_weight=0)
    """
    ups = torch.stack([s[1] for s in segments])    # [N, hidden]
    downs = torch.stack([s[2] for s in segments])  # [N, hidden]
    return ups, downs


def compute_assignment(dropped_segments, dropped_sources,
                       surviving_segments_by_expert, surviving_ids, threshold):
    """
    Assign dropped segments to the most compatible surviving expert.

    Returns:
      assignments: {surviving_expert_id: [(segment, source_expert_id)]}
      orphans: [(segment, source_expert_id, similarity)]
      n_assigned: int
      transfer_counts: {source_expert: {survivor_expert: count}}
    """
    if not dropped_segments:
        return ({eid: [] for eid in surviving_ids}, [], 0,
                defaultdict(lambda: defaultdict(int)))

    d_ups, d_downs = segments_to_matrix(dropped_segments)
    d_ups_norm = torch.nn.functional.normalize(d_ups, dim=1)
    d_downs_norm = torch.nn.functional.normalize(d_downs, dim=1)

    n_dropped = len(dropped_segments)
    best_sim = torch.full((n_dropped,), -1.0)
    best_expert = torch.full((n_dropped,), -1, dtype=torch.long)

    for eid in surviving_ids:
        s_segs = surviving_segments_by_expert[eid]
        s_ups, s_downs = segments_to_matrix(s_segs)
        s_ups_norm = torch.nn.functional.normalize(s_ups, dim=1)
        s_downs_norm = torch.nn.functional.normalize(s_downs, dim=1)

        up_sims = d_ups_norm @ s_ups_norm.T
        down_sims = d_downs_norm @ s_downs_norm.T
        combined = 0.5 * up_sims + 0.5 * down_sims
        max_sim_to_expert, _ = combined.max(dim=1)

        better = max_sim_to_expert > best_sim
        best_sim[better] = max_sim_to_expert[better]
        best_expert[better] = eid

    assignments = {eid: [] for eid in surviving_ids}
    orphans = []
    n_assigned = 0
    transfer_counts = defaultdict(lambda: defaultdict(int))

    for i in range(n_dropped):
        src = dropped_sources[i]
        if best_sim[i] >= threshold and best_expert[i] >= 0:
            dst = int(best_expert[i])
            assignments[dst].append((dropped_segments[i], src))
            transfer_counts[src][dst] += 1
            n_assigned += 1
        else:
            orphans.append((dropped_segments[i], src, float(best_sim[i])))

    return assignments, orphans, n_assigned, transfer_counts


def spherical_kmeans(segments, weights, k, max_iter=30, tol=1e-4):
    """
    Weighted spherical k-means on concatenated segment vectors.

    segments: list of (gate, up, down) tuples
    weights: importance weight per segment
    k: number of output centroids (= intermediate_size)

    Returns: k centroids as (gate, up, down) tuples
    """
    hidden = segments[0][0].shape[0]
    n = len(segments)

    # Build data matrix: concat gate, up, down -> [N, 3*hidden]
    data = torch.stack([torch.cat([s[0], s[1], s[2]]) for s in segments])  # [N, 3H]
    w = torch.tensor(weights, dtype=torch.float32)  # [N]

    if n <= k:
        # Fewer segments than target — pad with zeros
        result = []
        for i in range(n):
            result.append((segments[i][0].clone(), segments[i][1].clone(),
                           segments[i][2].clone()))
        for _ in range(k - n):
            result.append((torch.zeros(hidden), torch.zeros(hidden),
                           torch.zeros(hidden)))
        return result

    # Initialize centroids: top-k by activation magnitude
    magnitudes = torch.stack([s[0].abs().max() + s[0].abs().min().abs()
                              for s in segments])
    _, init_idx = magnitudes.topk(k)
    centroids = data[init_idx].clone()  # [k, 3H]
    centroids = torch.nn.functional.normalize(centroids, dim=1)

    for iteration in range(max_iter):
        # Assign: cosine similarity
        data_norm = torch.nn.functional.normalize(data, dim=1)
        sims = data_norm @ centroids.T  # [N, k]
        labels = sims.argmax(dim=1)     # [N]

        # Update centroids with norm equalization
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.sum() == 0:
                new_centroids[j] = centroids[j]
                continue

            cluster_data = data[mask]       # [C, 3H]
            cluster_w = w[mask]             # [C]

            # Norm equalization: normalize to mean norm within cluster
            norms = cluster_data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            mean_norm = norms.mean()
            equalized = cluster_data * (mean_norm / norms)

            # Weighted centroid
            cluster_w_norm = cluster_w / cluster_w.sum()
            centroid = (equalized * cluster_w_norm.unsqueeze(1)).sum(dim=0)
            new_centroids[j] = torch.nn.functional.normalize(centroid, dim=0) * mean_norm

        # Check convergence
        shift = (new_centroids - centroids).norm()
        centroids = new_centroids
        if shift < tol:
            break

    # Split centroids back into (gate, up, down)
    result = []
    for j in range(k):
        c = centroids[j]
        result.append((c[:hidden], c[hidden:2*hidden], c[2*hidden:]))

    return result


def build_residual_expert(orphans, expert_importances, drop_ids,
                          intermediate_size, hidden_size, kmeans_iters):
    """
    Build a residual expert from orphan segments that didn't match any survivor.
    If more than intermediate_size orphans, keep only the most important ones
    (from highest-contribution dropped experts), then cluster to intermediate_size.

    Returns: (gate_up, down) tensors or None if no orphans
    """
    if not orphans:
        return None, 0, 0, []

    # orphans are (segment_tuple, source_eid, similarity)
    # Keep orphans from highest-contribution dropped experts first
    n_total = len(orphans)
    # Sort by source expert importance descending
    orphans_sorted = sorted(
        orphans, key=lambda o: expert_importances[o[1]], reverse=True)

    if n_total > intermediate_size * 2:
        # Keep top 2x intermediate (k-means will compress to intermediate)
        kept_orphans = orphans_sorted[:intermediate_size * 2]
        n_discarded = n_total - len(kept_orphans)
    else:
        kept_orphans = orphans_sorted
        n_discarded = 0

    segments = [o[0] for o in kept_orphans]
    sources_kept = [o[1] for o in kept_orphans]

    # Weight orphans by their source expert importance
    weights = [expert_importances[s] for s in sources_kept]

    if len(segments) > intermediate_size:
        centroids = spherical_kmeans(
            segments, weights, intermediate_size, max_iter=kmeans_iters)
    else:
        centroids = [(s[0].clone(), s[1].clone(), s[2].clone())
                     for s in segments]
        while len(centroids) < intermediate_size:
            centroids.append((torch.zeros(hidden_size),
                              torch.zeros(hidden_size),
                              torch.zeros(hidden_size)))

    gate_rows = torch.stack([c[0] for c in centroids])
    up_rows = torch.stack([c[1] for c in centroids])
    down_cols = torch.stack([c[2] for c in centroids])

    gate_up = torch.cat([gate_rows, up_rows], dim=0)
    down = down_cols.T

    # Return: residual tensors, n_kept, n_discarded, list of source expert ids used
    return (gate_up, down), n_total - n_discarded, n_discarded, sources_kept


def process_layer(gate_up_proj, down_proj, keep_ids, drop_ids,
                  expert_importances, intermediate_size, hidden_size,
                  threshold, cluster_ratio, kmeans_iters):
    """
    Full DERN processing for one layer.

    gate_up_proj: [num_experts, 2*intermediate, hidden]
    down_proj: [num_experts, hidden, intermediate]

    Returns: modified gate_up_proj [keep_n+1, 2*intermediate, hidden],
             modified down_proj [keep_n+1, hidden, intermediate],
             stats dict
    """
    target_intermediate = int(intermediate_size * cluster_ratio)

    # Extract segments for all experts
    all_segments = {}
    for eid in list(keep_ids) + list(drop_ids):
        segs = extract_segments(
            gate_up_proj[eid].float(),
            down_proj[eid].float(),
            intermediate_size)
        all_segments[eid] = segs

    # Collect all dropped segments with source tracking
    all_dropped = []
    dropped_sources = []
    for eid in drop_ids:
        for seg in all_segments[eid]:
            all_dropped.append(seg)
            dropped_sources.append(eid)

    # Assign dropped segments to surviving experts
    surviving_segments = {eid: all_segments[eid] for eid in keep_ids}
    assignments, orphans, n_assigned, transfer_counts = compute_assignment(
        all_dropped, dropped_sources, surviving_segments, keep_ids, threshold)

    # For each surviving expert: cluster original + transferred segments
    new_gate_up_list = []
    new_down_list = []

    for eid in sorted(keep_ids):
        original = all_segments[eid]
        transferred_with_src = assignments[eid]
        transferred = [t[0] for t in transferred_with_src]
        all_segs = original + transferred

        imp = expert_importances[eid]
        weights = [imp] * len(original)
        for seg in transferred:
            weights.append(imp * 0.5)

        if len(all_segs) > target_intermediate:
            centroids = spherical_kmeans(
                all_segs, weights, target_intermediate,
                max_iter=kmeans_iters)
        else:
            centroids = [(s[0].clone(), s[1].clone(), s[2].clone())
                         for s in all_segs]
            while len(centroids) < target_intermediate:
                centroids.append((torch.zeros(hidden_size),
                                  torch.zeros(hidden_size),
                                  torch.zeros(hidden_size)))

        gate_rows = torch.stack([c[0] for c in centroids])
        up_rows = torch.stack([c[1] for c in centroids])
        down_cols = torch.stack([c[2] for c in centroids])

        new_gate_up = torch.cat([gate_rows, up_rows], dim=0)
        new_down = down_cols.T

        new_gate_up_list.append(new_gate_up)
        new_down_list.append(new_down)

    # Build residual expert from orphan segments
    residual, n_residual_kept, n_residual_discarded, residual_sources = \
        build_residual_expert(
            orphans, expert_importances, drop_ids,
            target_intermediate, hidden_size, kmeans_iters)

    if residual is not None:
        new_gate_up_list.append(residual[0])
        new_down_list.append(residual[1])

    new_gate_up_proj = torch.stack(new_gate_up_list)
    new_down_proj = torch.stack(new_down_list)

    stats = {
        "n_assigned": n_assigned,
        "n_orphans": len(orphans),
        "n_residual_kept": n_residual_kept,
        "n_residual_discarded": n_residual_discarded,
        "has_residual": residual is not None,
    }
    transfer_counts_dict = {
        int(src): {int(dst): cnt for dst, cnt in dsts.items()}
        for src, dsts in transfer_counts.items()
    }
    return (new_gate_up_proj, new_down_proj, stats,
            transfer_counts_dict, residual_sources)


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
    target_intermediate = int(intermediate_size * args.cluster_ratio)

    print(f"Model: {source_dir}")
    print(f"  Layers: {num_layers}, Experts: {num_experts}, "
          f"Intermediate: {intermediate_size}, Hidden: {hidden_size}")
    print(f"  Target: {target_experts} experts, "
          f"intermediate {target_intermediate}")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print(f"  K-means iters: {args.kmeans_iters}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = source_dir.parent / f"gemma-4-A4B-{target_experts}e-dern"
    print(f"  Output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load contribution data
    if args.contribution_file:
        contrib_path = args.contribution_file
    else:
        contrib_path = source_dir / "eval_results" / "expert_contributions_full.json"
    print(f"\nLoading contributions from {contrib_path}...")
    contributions = load_contributions(contrib_path, num_layers, num_experts)

    # Select experts
    keep_map, drop_map = select_experts(
        contributions, num_layers, num_experts, target_experts)

    # Print per-layer summary
    print(f"\nPer-layer contribution retention:")
    for li in range(num_layers):
        total = sum(contributions[li])
        kept = sum(contributions[li][e] for e in keep_map[li])
        pct = kept / max(total, 1e-10) * 100
        print(f"  L{li:2d}: {pct:.2f}% retained, "
              f"drop {len(drop_map[li])} experts")

    # Load safetensors index
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

    # Pass 1: DERN processing per layer (heavy computation)
    actual_experts = target_experts + 1  # +1 for residual expert
    gate_up_cache = {}
    down_cache = {}
    layer_stats = {}
    total_assigned = 0
    total_orphans = 0
    total_discarded = 0

    print(f"\nPass 1: DERN processing {num_layers} layers...")
    print(f"  Output: {target_experts} kept + 1 residual = {actual_experts} experts")

    transfer_counts_per_layer = {}
    residual_sources_per_layer = {}

    for li in tqdm(range(num_layers), desc="DERN"):
        gu_key = f"model.language_model.layers.{li}.experts.gate_up_proj"
        dn_key = f"model.language_model.layers.{li}.experts.down_proj"

        gate_up = shard_files[weight_map[gu_key]].get_tensor(gu_key)
        down = shard_files[weight_map[dn_key]].get_tensor(dn_key)

        t0 = time.time()
        new_gu, new_dn, stats, transfer_counts, residual_srcs = process_layer(
            gate_up, down,
            keep_map[li], drop_map[li],
            contributions[li],
            intermediate_size, hidden_size,
            args.similarity_threshold,
            args.cluster_ratio,
            args.kmeans_iters)
        elapsed = time.time() - t0

        gate_up_cache[li] = new_gu.to(torch.bfloat16)
        down_cache[li] = new_dn.to(torch.bfloat16)
        layer_stats[li] = stats
        transfer_counts_per_layer[li] = transfer_counts
        residual_sources_per_layer[li] = residual_srcs
        total_assigned += stats["n_assigned"]
        total_orphans += stats["n_orphans"]
        total_discarded += stats["n_residual_discarded"]

        res = "+" if stats["has_residual"] else ""
        tqdm.write(
            f"  L{li:2d}: {stats['n_assigned']} assigned, "
            f"{stats['n_orphans']} orphans{res}, "
            f"{stats['n_residual_discarded']} final discard, "
            f"{elapsed:.0f}s")

    # Pass 2: Write all tensors
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

            # === Expert gate_up_proj: use cached DERN result ===
            m = re.match(
                r"model\.language_model\.layers\.(\d+)\.experts\.gate_up_proj",
                key)
            if m:
                tensor = gate_up_cache.pop(int(m.group(1)))

            # === Expert down_proj: use cached DERN result ===
            m = re.match(
                r"model\.language_model\.layers\.(\d+)\.experts\.down_proj",
                key)
            if m:
                tensor = down_cache.pop(int(m.group(1)))

            # === Router proj.weight: keep survivors unchanged + residual row ===
            m = re.match(
                r"model\.language_model\.layers\.(\d+)\.router\.proj\.weight",
                key)
            if m:
                li = int(m.group(1))
                orig = tensor.float()

                # Surviving experts' routers stay unchanged (don't disturb routing)
                kept_rows = orig[sorted(keep_map[li])]

                # Residual: weighted average of source routers proportional to
                # how many of each source's neurons ended up in the residual
                res_srcs = residual_sources_per_layer[li]
                if res_srcs:
                    from collections import Counter
                    src_counts = Counter(res_srcs)
                    total_res = sum(src_counts.values())
                    residual_row = torch.zeros_like(orig[0])
                    for src, cnt in src_counts.items():
                        residual_row += orig[src] * (cnt / total_res)
                    residual_row = residual_row.unsqueeze(0)
                else:
                    residual_row = torch.zeros((1, orig.shape[1]),
                                               dtype=orig.dtype)

                tensor = torch.cat([kept_rows, residual_row], dim=0).to(
                    torch.bfloat16)

            # === Router per_expert_scale: keep survivors + weighted residual ===
            m = re.match(
                r"model\.language_model\.layers\.(\d+)\.router\.per_expert_scale",
                key)
            if m:
                li = int(m.group(1))
                kept = tensor[sorted(keep_map[li])]
                res_srcs = residual_sources_per_layer[li]
                if res_srcs:
                    from collections import Counter
                    src_counts = Counter(res_srcs)
                    total_res = sum(src_counts.values())
                    residual_scale = sum(
                        tensor[src].float() * (cnt / total_res)
                        for src, cnt in src_counts.items())
                    residual_scale = residual_scale.unsqueeze(0).to(tensor.dtype)
                else:
                    residual_scale = torch.zeros(1, dtype=tensor.dtype)
                tensor = torch.cat([kept, residual_scale])

            # === Write to output shard ===
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

    # Rename with total count
    for old_idx in range(1, shard_idx + 1):
        old_name = output_dir / f"model-{old_idx:05d}.safetensors"
        new_name = output_dir / f"model-{old_idx:05d}-of-{shard_idx:05d}.safetensors"
        old_name.rename(new_name)
        for k, v in new_weight_map.items():
            if v == f"model-{old_idx:05d}.safetensors":
                new_weight_map[k] = new_name.name

    for sf in shard_files.values():
        del sf

    # Write index
    new_index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # Update config
    new_config = json.loads(json.dumps(config))
    new_config["text_config"]["num_experts"] = actual_experts
    if args.cluster_ratio != 1.0:
        new_config["text_config"]["moe_intermediate_size"] = target_intermediate
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
        "base_model": str(source_dir),
        "method": "DERN",
        "original_experts": num_experts,
        "target_experts": target_experts,
        "actual_experts": actual_experts,
        "residual_expert_idx": actual_experts - 1,
        "intermediate_size": target_intermediate,
        "similarity_threshold": args.similarity_threshold,
        "cluster_ratio": args.cluster_ratio,
        "kmeans_iters": args.kmeans_iters,
        "total_segments_assigned": total_assigned,
        "total_orphans_to_residual": total_orphans,
        "total_segments_discarded": total_discarded,
        "per_layer_stats": {str(li): layer_stats[li] for li in range(num_layers)},
        "per_layer_keep": {str(li): keep_map[li] for li in range(num_layers)},
        "per_layer_drop": {str(li): drop_map[li] for li in range(num_layers)},
        "total_params_bf16": total_size / 2,
        "total_size_gb": total_size / (1024 ** 3),
    }
    with open(output_dir / "dern_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Size: {total_size / 1024**3:.1f} GB "
          f"({total_size / 2 / 1e9:.1f}B params @ bf16)")
    print(f"  Shards: {shard_idx}")
    print(f"  Experts: {actual_experts} ({target_experts} kept + 1 residual)")
    print(f"  Segments assigned to survivors: {total_assigned}")
    print(f"  Segments to residual expert: {total_orphans}")
    print(f"  Segments discarded: {total_discarded}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
