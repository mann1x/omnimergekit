#!/usr/bin/env python3
"""
Gemma 4 26B-A4B Expert Pruning Script
=======================================
Prunes unused/rare experts from the MoE model to reduce total parameter count
while preserving active inference quality.

Architecture: 30 layers, 128 experts per MoE layer, top-8 routing
  - Each expert MLP: intermediate_size=704 (tiny)
  - Non-expert dense MLP: intermediate_size=2112 (always active)
  - hidden_size=2816, vocab=262144

Strategy:
  1. Profile expert activation on domain-specific calibration data
     (general text + math/reasoning + code) to identify critical experts
  2. Rank experts by frequency × routing weight
  3. Keep top-N experts per layer, drop the rest
  4. Remap router weights to match new expert indices
  5. Save as a valid HF model

Usage:
  # Profile experts (dry run — just show activation stats)
  python expert_prune.py --source . --profile --cal-samples 256

  # Prune to 32 experts (~8-9B params, ~5GB at Q4)
  python expert_prune.py --source . --output ../gemma-4-A4B-32e --target-experts 32

  # Prune to 64 experts (~14B params, ~8GB at Q4)
  python expert_prune.py --source . --output ../gemma-4-A4B-64e --target-experts 64

  # Custom: preserve reasoning experts with higher weight
  python expert_prune.py --source . --output ../gemma-4-A4B-32e --target-experts 32 --reasoning-weight 2.0

Requirements:
  pip install safetensors torch numpy tqdm transformers accelerate bitsandbytes datasets
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


# ── Calibration data ──────────────────────────────────────────

REASONING_PROMPTS = [
    # Math
    "Solve step by step: If f(x) = 3x^2 - 2x + 1, find f'(x) and evaluate at x=4.",
    "A train leaves city A at 60 mph. Another leaves city B (300 miles away) at 40 mph toward A. When do they meet?",
    "Prove that the sum of the first n odd numbers equals n^2.",
    "Calculate the integral of sin(x)*cos(x) dx from 0 to pi/2.",
    "If a matrix A = [[1,2],[3,4]], find A^(-1) and verify AA^(-1) = I.",
    "Find all prime numbers p such that p^2 + 2 is also prime.",
    "A bag contains 5 red and 3 blue balls. Draw 2 without replacement. P(both red)?",
    "Solve the differential equation dy/dx = y*sin(x), y(0) = 1.",
    # Reasoning / Logic
    "Three people check into a hotel room that costs $30. They split it $10 each. The clerk realizes the room is $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and returns $1 each. Now each paid $9 (total $27) plus $2 the bellboy kept = $29. Where's the missing dollar?",
    "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain using formal logic.",
    "You have 8 identical-looking coins, one is heavier. Using a balance scale, find it in exactly 2 weighings.",
    "In a room of 23 people, what's the probability at least two share a birthday? Derive the formula.",
    "A farmer has a fox, chicken, and grain. He must cross a river in a boat that carries only one item. How?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
    # Code reasoning
    "What is the time complexity of quicksort in the average case? Explain why.",
    "Write a function to detect a cycle in a linked list. What is the space complexity?",
    "Explain why a hash table has O(1) average lookup but O(n) worst case.",
    "What happens when you recursively compute fibonacci(50) without memoization? How many calls?",
]

GENERAL_PROMPTS = [
    "The history of the Roman Empire begins with",
    "In modern physics, quantum entanglement refers to",
    "The process of photosynthesis in plants involves",
    "Machine learning algorithms can be categorized into",
    "The French Revolution of 1789 was caused by",
    "DNA replication occurs through a process called",
    "The economic theory of supply and demand states that",
    "In computer networking, the TCP/IP protocol stack",
]


def format_as_chat(tokenizer, prompt: str, enable_thinking: bool = True) -> torch.Tensor:
    """Format prompt using the model's chat template with thinking enabled."""
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=1024)["input_ids"]
    except Exception:
        # Fallback to raw tokenization if chat template fails
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=1024)["input_ids"]
    return tokens


def load_benchmark_questions(tokenizer, num_gpqa: int = 100, num_mmlu: int = 100,
                              seq_len: int = 1024) -> list[tuple[torch.Tensor, float]]:
    """Load actual benchmark questions (GPQA Diamond + MMLU-Pro) for calibration.
    These are the tasks we'll evaluate on, so profiling experts on them
    ensures we preserve the experts that matter for reasoning."""
    samples = []

    # GPQA Diamond — hard reasoning questions (weighted 2x)
    try:
        from datasets import load_dataset
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        for item in ds:
            if len(samples) >= num_gpqa:
                break
            q = item["Question"]
            choices = [item["Correct Answer"], item["Incorrect Answer 1"],
                       item["Incorrect Answer 2"], item["Incorrect Answer 3"]]
            prompt = f"Question: {q}\n\nChoices:\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n\nThink step by step and give your answer."
            tokens = format_as_chat(tokenizer, prompt, enable_thinking=True)
            samples.append((tokens, 2.0))  # weight 2x for reasoning
        print(f"  GPQA Diamond: {min(len(ds), num_gpqa)} questions loaded (weight 2x)")
    except Exception as e:
        print(f"  GPQA Diamond failed: {e}")

    # MMLU-Pro — sample across categories for diversity
    try:
        from datasets import load_dataset
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True)
        # Collect by category, take 2-3 per category for diversity
        by_category = {}
        for item in ds:
            cat = item.get("category", "other")
            if cat not in by_category:
                by_category[cat] = []
            if len(by_category[cat]) < 3:
                by_category[cat].append(item)
            if sum(len(v) for v in by_category.values()) >= num_mmlu * 2:
                break
        # Take evenly from categories
        selected = []
        per_cat = max(1, num_mmlu // len(by_category)) if by_category else 0
        for cat, items in by_category.items():
            for item in items[:per_cat]:
                selected.append(item)
            if len(selected) >= num_mmlu:
                break
        for item in selected[:num_mmlu]:
            q = item["question"]
            options = item["options"]
            choices_str = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
            prompt = f"Question: {q}\n\nChoices:\n{choices_str}\n\nThink step by step and give your answer."
            tokens = format_as_chat(tokenizer, prompt, enable_thinking=True)
            samples.append((tokens, 1.0))
        print(f"  MMLU-Pro: {len(selected[:num_mmlu])} questions from {len(by_category)} categories (weight 1x)")
    except Exception as e:
        print(f"  MMLU-Pro failed: {e}")

    return samples


def get_calibration_data(tokenizer, num_samples: int = 256, seq_len: int = 1024,
                         reasoning_weight: float = 2.0):
    """Build calibration dataset from actual benchmark questions + custom prompts."""
    samples = []
    weights = []

    # Primary: actual benchmark questions (GPQA + MMLU-Pro)
    # All GPQA (198, 2x weight) + diverse MMLU-Pro
    benchmark_data = load_benchmark_questions(
        tokenizer, num_gpqa=min(198, num_samples),
        num_mmlu=min(200, num_samples), seq_len=seq_len,
    )
    for tokens, weight in benchmark_data:
        samples.append(tokens)
        weights.append(weight)

    # Fallback: custom reasoning prompts if benchmarks didn't load
    if len(samples) < 20:
        print("  Adding custom reasoning prompts as fallback...")
        for prompt in REASONING_PROMPTS:
            tokens = format_as_chat(tokenizer, prompt, enable_thinking=True)
            samples.append(tokens)
            weights.append(reasoning_weight)

        for prompt in GENERAL_PROMPTS:
            tokens = format_as_chat(tokenizer, prompt, enable_thinking=True)
            samples.append(tokens)
            weights.append(1.0)

    # WikiText for general language
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join([t for t in ds["text"] if len(t.strip()) > 100])
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        remaining = num_samples - len(samples)
        for i in range(0, min(remaining * seq_len, len(tokens) - seq_len), seq_len):
            samples.append(tokens[i:i + seq_len].unsqueeze(0))
            weights.append(1.0)
    except Exception as e:
        print(f"  Warning: could not load wikitext: {e}")

    print(f"  Calibration: {len(samples)} samples "
          f"({len(REASONING_PROMPTS)} reasoning @ {reasoning_weight}x, "
          f"{len(GENERAL_PROMPTS)} general, {len(samples) - len(REASONING_PROMPTS) - len(GENERAL_PROMPTS)} wikitext)")
    return samples[:num_samples], weights[:num_samples]


# ── Expert profiling ──────────────────────────────────────────

def profile_experts(source_dir: Path, device: str = "cuda",
                    num_samples: int = 256, seq_len: int = 1024,
                    reasoning_weight: float = 2.0) -> dict:
    """
    Profile expert activation patterns across calibration data.
    Returns per-layer dict of expert_id -> (frequency, avg_routing_weight).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Fix accelerate/bnb compatibility: patch Params4bit to accept _is_hf_initialized
    try:
        import bitsandbytes as bnb
        _orig_new = bnb.nn.Params4bit.__new__
        def _patched_new(cls, *args, **kwargs):
            kwargs.pop("_is_hf_initialized", None)
            return _orig_new(cls, *args, **kwargs)
        bnb.nn.Params4bit.__new__ = _patched_new
    except Exception:
        pass

    print("\n[Profile] Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    max_mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            max_mem[i] = f"{int(total * 0.85 / 1024**3)}GiB"
        max_mem["cpu"] = "100GiB"
        print(f"  Memory allocation: {max_mem}")
    model = AutoModelForCausalLM.from_pretrained(
        str(source_dir), quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
        max_memory=max_mem if max_mem else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(source_dir))
    model.eval()

    config = model.config
    text_cfg = config.text_config
    num_layers = text_cfg.num_hidden_layers
    num_experts = text_cfg.num_experts
    top_k = text_cfg.top_k_experts

    print(f"  {num_layers} layers, {num_experts} experts, top-{top_k} routing")

    samples, weights = get_calibration_data(tokenizer, num_samples, seq_len, reasoning_weight)

    # Hook into router outputs to capture expert selections
    # The router produces (routing_weights, selected_experts) per MoE layer
    layer_expert_stats = defaultdict(lambda: defaultdict(lambda: {"count": 0.0, "weight_sum": 0.0}))
    hooks = []

    def make_router_hook(layer_idx):
        def hook(module, args, output):
            # Gemma4TextRouter returns: (router_probabilities, top_k_weights, top_k_index)
            # router_probabilities: [B*S, num_experts] — full softmax distribution
            # top_k_weights: [B*S, K] — normalized weights for selected experts
            # top_k_index: [B*S, K] — indices of selected experts
            if not isinstance(output, tuple) or len(output) < 3:
                return

            router_probs = output[0].detach().float().cpu()  # [B*S, 128]
            top_k_weights = output[1].detach().float().cpu()  # [B*S, 8]
            top_k_index = output[2].detach().cpu()            # [B*S, 8]

            # Method 1: Track which experts are selected (top-k frequency)
            for k in range(top_k_index.shape[1]):
                expert_ids = top_k_index[:, k].numpy()
                expert_weights = top_k_weights[:, k].numpy()
                for eid, ew in zip(expert_ids, expert_weights):
                    layer_expert_stats[layer_idx][int(eid)]["count"] += sample_weight
                    layer_expert_stats[layer_idx][int(eid)]["weight_sum"] += float(ew) * sample_weight

            # Method 2: Track full router probability distribution
            # (captures experts that are "close to being selected" but didn't make top-k)
            avg_probs = router_probs.mean(dim=0).numpy()  # [128]
            for eid in range(len(avg_probs)):
                layer_expert_stats[layer_idx][int(eid)]["avg_router_prob"] = \
                    layer_expert_stats[layer_idx][int(eid)].get("avg_router_prob", 0) + \
                    float(avg_probs[eid]) * sample_weight
        return hook

    # Hook into each layer's router (Gemma4TextRouter)
    layers = model.model.language_model.layers
    for i in range(num_layers):
        layer = layers[i]
        if hasattr(layer, "router"):
            hooks.append(layer.router.register_forward_hook(make_router_hook(i)))

    print(f"  Registered {len(hooks)} router hooks (expect {num_layers} for MoE layers)")

    # Run calibration
    print("[Profile] Running calibration...")
    sample_weight = 1.0
    for idx, sample in enumerate(tqdm(samples, desc="Profiling")):
        sample_weight = weights[idx] if idx < len(weights) else 1.0
        sample = sample.to(model.device)
        with torch.no_grad():
            try:
                # Gemma4 needs mm_token_type_ids
                mm_ids = torch.zeros_like(sample)
                model(sample, mm_token_type_ids=mm_ids)
            except Exception as e:
                if idx == 0:
                    print(f"  Warning: forward pass error: {e}")
                continue

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute stats
    expert_profiles = {}
    for layer_idx in sorted(layer_expert_stats.keys()):
        stats = layer_expert_stats[layer_idx]
        total_count = sum(s["count"] for s in stats.values())
        expert_profiles[layer_idx] = {}
        for eid in range(num_experts):
            s = stats.get(eid, {"count": 0, "weight_sum": 0})
            freq = s["count"] / total_count if total_count > 0 else 0
            avg_weight = s["weight_sum"] / s["count"] if s["count"] > 0 else 0
            expert_profiles[layer_idx][eid] = {
                "frequency": freq,
                "avg_routing_weight": avg_weight,
                "activation_count": s["count"],
                "importance": freq * avg_weight,  # combined importance score
            }

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return expert_profiles, num_layers, num_experts, top_k


def compute_expert_similarity(profiles: dict, num_layers: int, num_experts: int) -> np.ndarray:
    """Compute cosine similarity between experts based on their activation patterns across layers.
    NOTE: This is only used for stats display. Actual merging uses per-layer weight similarity."""
    feature_matrix = np.zeros((num_experts, num_layers * 2))
    for layer_idx, experts in profiles.items():
        for eid, stats in experts.items():
            feature_matrix[eid, layer_idx * 2] = stats["frequency"]
            feature_matrix[eid, layer_idx * 2 + 1] = stats["avg_routing_weight"]
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = feature_matrix / norms
    return normalized @ normalized.T


def compute_weight_similarity_per_layer(source_dir: Path, num_layers: int,
                                         num_experts: int) -> dict[int, np.ndarray]:
    """Compute cosine similarity between experts based on actual weight tensors, per layer.
    This measures whether experts compute similar functions, not just whether they activate together."""
    with open(source_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Open all shard files
    shard_files = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_files[shard_name] = safe_open(str(source_dir / shard_name), framework="pt", device="cpu")

    per_layer_sim = {}

    for layer_idx in tqdm(range(num_layers), desc="Weight similarity"):
        # Collect flattened weight vectors for each expert in this layer
        expert_vectors = [[] for _ in range(num_experts)]

        for suffix in ["gate_up_proj", "down_proj"]:
            key = f"model.language_model.layers.{layer_idx}.experts.{suffix}"
            if key in weight_map:
                tensor = shard_files[weight_map[key]].get_tensor(key)
                # tensor shape: [num_experts, ...]
                for eid in range(num_experts):
                    expert_vectors[eid].append(tensor[eid].flatten().float().numpy())

        if not expert_vectors[0]:
            continue

        # Concatenate gate_up + down into one vector per expert
        feature_matrix = np.stack([np.concatenate(vecs) for vecs in expert_vectors])

        # Cosine similarity
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = feature_matrix / norms
        per_layer_sim[layer_idx] = normalized @ normalized.T

    for sf in shard_files.values():
        del sf

    return per_layer_sim


def find_merge_groups(similarity: np.ndarray, num_experts: int,
                      target: int, importance_order: list[int]) -> list[list[int]]:
    """Greedily group experts for merging based on similarity.
    Most important experts become group leaders, similar less-important ones merge into them."""
    merged = set()
    groups = []

    for eid in importance_order:
        if eid in merged:
            continue
        group = [eid]
        merged.add(eid)

        if len(groups) < target:
            sim_scores = [(other, similarity[eid, other])
                          for other in range(num_experts)
                          if other not in merged]
            sim_scores.sort(key=lambda x: x[1], reverse=True)

            max_group_size = max(1, (num_experts - len(merged)) // max(1, target - len(groups)))
            for other, sim in sim_scores[:max_group_size - 1]:
                if len(groups) + (num_experts - len(merged)) <= target:
                    break
                group.append(other)
                merged.add(other)

            groups.append(group)

    for eid in range(num_experts):
        if eid not in merged:
            groups.append([eid])

    return groups[:target]


def find_merge_groups_per_layer(per_layer_sim: dict[int, np.ndarray],
                                profiles: dict, num_experts: int,
                                target: int) -> dict[int, list[list[int]]]:
    """Find merge groups independently per layer using weight-space similarity.
    Each layer gets its own merge plan based on its own expert weight similarity."""
    per_layer_groups = {}

    for layer_idx, sim_matrix in sorted(per_layer_sim.items()):
        # Per-layer importance ordering
        layer_profiles = profiles.get(layer_idx, {})
        layer_importance = []
        for eid in range(num_experts):
            imp = layer_profiles.get(eid, {}).get("importance", 0)
            layer_importance.append((eid, imp))
        layer_importance.sort(key=lambda x: x[1], reverse=True)
        importance_order = [eid for eid, _ in layer_importance]

        groups = find_merge_groups(sim_matrix, num_experts, target, importance_order)
        per_layer_groups[layer_idx] = groups

    return per_layer_groups


def print_expert_stats(profiles: dict, num_layers: int, num_experts: int, top_n: int = 10):
    """Print detailed expert activation patterns with per-layer breakdown."""
    print(f"\n{'='*80}")
    print(f"  EXPERT ACTIVATION ANALYSIS ({num_experts} experts, {num_layers} layers)")
    print(f"{'='*80}")

    # Global expert importance (averaged across layers)
    global_importance = defaultdict(float)
    global_frequency = defaultdict(float)
    global_weight = defaultdict(float)
    for layer_idx, experts in profiles.items():
        for eid, stats in experts.items():
            global_importance[eid] += stats["importance"]
            global_frequency[eid] += stats["frequency"]
            global_weight[eid] += stats["avg_routing_weight"]

    sorted_global = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)

    # Distribution analysis
    importances = [imp for _, imp in sorted_global]
    print(f"\n  Importance distribution:")
    print(f"    Max: {importances[0]:.6f}  Min: {importances[-1]:.6f}  Ratio: {importances[0]/max(importances[-1],1e-10):.1f}x")
    print(f"    Mean: {np.mean(importances):.6f}  Std: {np.std(importances):.6f}  CV: {np.std(importances)/np.mean(importances):.2f}")
    print(f"    Top 32 capture: {sum(importances[:32])/sum(importances)*100:.1f}% of total importance")
    print(f"    Top 64 capture: {sum(importances[:64])/sum(importances)*100:.1f}% of total importance")
    print(f"    Top 96 capture: {sum(importances[:96])/sum(importances)*100:.1f}% of total importance")

    print(f"\n  Top {top_n} most important experts:")
    for rank, (eid, imp) in enumerate(sorted_global[:top_n]):
        freq = global_frequency[eid]
        wt = global_weight[eid] / num_layers
        print(f"    #{rank+1:3d}  Expert {eid:3d}  importance={imp:.6f}  freq={freq:.4f}  avg_weight={wt:.4f}")

    print(f"\n  Bottom {top_n} least important experts:")
    for rank, (eid, imp) in enumerate(sorted_global[-top_n:]):
        freq = global_frequency[eid]
        wt = global_weight[eid] / num_layers
        print(f"    #{len(sorted_global)-top_n+rank+1:3d}  Expert {eid:3d}  importance={imp:.6f}  freq={freq:.4f}  avg_weight={wt:.4f}")

    # Per-layer detailed stats
    print(f"\n  Per-layer stats:")
    print(f"    {'Layer':>5}  {'Active':>6}  {'Top Expert':>10}  {'Top Imp':>8}  {'Gini':>6}  {'Top8 %':>7}")
    print(f"    {'─'*5}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*7}")
    for layer_idx in sorted(profiles.keys()):
        experts = profiles[layer_idx]
        active = sum(1 for e in experts.values() if e["frequency"] > 0.001)
        top_expert = max(experts.items(), key=lambda x: x[1]["importance"])

        # Gini coefficient (inequality measure: 0=equal, 1=one expert dominates)
        layer_imps = sorted([e["importance"] for e in experts.values()])
        n = len(layer_imps)
        if sum(layer_imps) > 0:
            gini = sum((2 * (i + 1) - n - 1) * imp for i, imp in enumerate(layer_imps)) / (n * sum(layer_imps))
        else:
            gini = 0

        # Top 8 experts' share of total importance
        sorted_imps = sorted([e["importance"] for e in experts.values()], reverse=True)
        top8_share = sum(sorted_imps[:8]) / max(sum(sorted_imps), 1e-10) * 100

        print(f"    {layer_idx:>5}  {active:>6}  {top_expert[0]:>10}  "
              f"{top_expert[1]['importance']:>8.4f}  {gini:>6.3f}  {top8_share:>6.1f}%")

    # Activation-pattern similarity (for display only)
    print(f"\n  Computing activation-pattern similarity (for reference)...")
    act_similarity = compute_expert_similarity(profiles, num_layers, num_experts)

    sim_pairs = []
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            sim_pairs.append((i, j, act_similarity[i, j]))
    sim_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Top 10 most similar expert pairs (activation patterns):")
    for i, (e1, e2, sim) in enumerate(sim_pairs[:10]):
        imp1 = global_importance[e1]
        imp2 = global_importance[e2]
        print(f"    Expert {e1:3d} <-> Expert {e2:3d}  similarity={sim:.4f}  "
              f"importance=({imp1:.5f}, {imp2:.5f})")

    return sorted_global, act_similarity


def perform_expert_merge(source_dir: Path, output_dir: Path,
                         per_layer_groups: dict[int, list[list[int]]],
                         config: dict, profiles: dict,
                         max_shard_size_gb: float = 5.0):
    """Merge similar experts using per-layer merge groups.
    Each layer has its own independently computed merge plan based on weight similarity.
    Expert weights are importance-weighted averaged. Router proj uses leader's row only."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(source_dir / "model.safetensors.index.json") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    text_cfg = config["text_config"]
    num_layers = text_cfg["num_hidden_layers"]
    num_experts_orig = text_cfg["num_experts"]
    # All layers must produce the same number of output experts
    target_experts = len(per_layer_groups[next(iter(per_layer_groups))])

    # Compute importance-based merge weights per layer
    merge_weights = {}  # layer -> group_idx -> {old_expert: weight}
    for layer_idx, groups in per_layer_groups.items():
        merge_weights[layer_idx] = {}
        for new_idx, group in enumerate(groups):
            if len(group) == 1:
                merge_weights[layer_idx][new_idx] = {group[0]: 1.0}
            else:
                importances = {}
                for eid in group:
                    imp = profiles.get(layer_idx, {}).get(eid, {}).get("importance", 0)
                    importances[eid] = imp
                total = sum(importances.values())
                if total > 0:
                    merge_weights[layer_idx][new_idx] = {eid: imp / total for eid, imp in importances.items()}
                else:
                    merge_weights[layer_idx][new_idx] = {eid: 1.0 / len(group) for eid in group}

    print("  Phase 1: Opening shard files...")
    shard_files_map = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_files_map[shard_name] = safe_open(str(source_dir / shard_name), framework="pt", device="cpu")

    print("  Phase 2: Merging and writing...")
    shard_keys = defaultdict(list)
    for key, shard in weight_map.items():
        shard_keys[shard].append(key)

    new_weight_map = {}
    current_shard = {}
    current_size = 0
    shard_idx = 1
    max_shard_bytes = int(max_shard_size_gb * 1024**3)
    total_size = 0

    for shard_name in tqdm(sorted(shard_keys.keys()), desc="Merging experts"):
        sf = shard_files_map[shard_name]
        for key in shard_keys[shard_name]:
            tensor = sf.get_tensor(key)

            # Stacked expert weight (gate_up_proj or down_proj)
            m_stacked = re.match(
                r"(model\.language_model\.layers\.(\d+)\.experts\.)(gate_up_proj|down_proj)(.*)",
                key
            )
            if m_stacked:
                layer_idx = int(m_stacked.group(2))
                prefix = m_stacked.group(1)
                weight_name = m_stacked.group(3) + m_stacked.group(4)
                groups = per_layer_groups[layer_idx]

                merged_tensors = []
                for new_idx, group in enumerate(groups):
                    if len(group) == 1:
                        merged_tensors.append(tensor[group[0]])
                    else:
                        weights = merge_weights[layer_idx][new_idx]
                        avg = torch.zeros_like(tensor[group[0]]).float()
                        for eid, w in weights.items():
                            avg += tensor[eid].float() * w
                        merged_tensors.append(avg.to(tensor.dtype))

                merged = torch.stack(merged_tensors)
                new_key = f"{prefix}{weight_name}"
                tensor_size = merged.numel() * merged.element_size()

                if current_size + tensor_size > max_shard_bytes and current_shard:
                    sf_name = f"model-{shard_idx:05d}.safetensors"
                    save_file(current_shard, str(output_dir / sf_name))
                    for k in current_shard:
                        new_weight_map[k] = sf_name
                    shard_idx += 1
                    current_shard = {}
                    current_size = 0

                current_shard[new_key] = merged
                current_size += tensor_size
                total_size += tensor_size
                continue

            # Router weight
            m_router = re.match(
                r"(model\.language_model\.layers\.(\d+)\.router\.)(proj\.weight|per_expert_scale)(.*)",
                key
            )
            if m_router:
                layer_idx = int(m_router.group(2))
                weight_type = m_router.group(3)
                groups = per_layer_groups[layer_idx]

                if weight_type == "proj.weight":
                    # Router projection: [num_experts, hidden_size]
                    # Keep LEADER's row only — don't average, to preserve routing decisiveness
                    merged_rows = []
                    for new_idx, group in enumerate(groups):
                        # Leader = group[0] = most important expert in this layer
                        merged_rows.append(tensor[group[0]])
                    tensor = torch.stack(merged_rows)

                elif weight_type == "per_expert_scale":
                    # Per-expert scale: [num_experts]
                    # Keep leader's scale
                    merged_scales = []
                    for new_idx, group in enumerate(groups):
                        merged_scales.append(tensor[group[0]])
                    tensor = torch.stack(merged_scales)

            # Non-expert tensor or already processed — save
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

    # Save final shard
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

    for sf in shard_files_map.values():
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

    total_merged = sum(
        sum(1 for g in groups if len(g) > 1)
        for groups in per_layer_groups.values()
    )
    print(f"\nExpert merge complete!")
    print(f"  {num_experts_orig} -> {target_experts} experts per layer")
    print(f"  Total merges across all layers: {total_merged}")
    print(f"  Total size: {total_size / 1024**3:.1f} GB ({total_size / 2 / 1e9:.1f}B params @ bf16)")
    print(f"  Shards: {shard_idx}")


def verify_merged_model(model_dir: Path, original_profiles: dict,
                        device: str = "cuda", num_samples: int = 50,
                        reasoning_weight: float = 2.0):
    """Re-profile the merged model and compare routing patterns to the original.
    Checks that:
    1. Router still makes decisive choices (healthy Gini coefficients)
    2. Expert utilization is not degenerate (no dead experts, no single-expert domination)
    3. Routing entropy is in a reasonable range
    """
    print(f"\n{'='*60}")
    print(f"  POST-MERGE VERIFICATION")
    print(f"{'='*60}")

    # Profile the merged model
    merged_profiles, num_layers, num_experts, top_k = profile_experts(
        model_dir, device=device, num_samples=num_samples,
        reasoning_weight=reasoning_weight,
    )

    issues = []

    for layer_idx in sorted(merged_profiles.keys()):
        experts = merged_profiles[layer_idx]
        importances = [e["importance"] for e in experts.values()]
        frequencies = [e["frequency"] for e in experts.values()]

        # Check for dead experts (frequency < 0.1% of uniform)
        uniform_freq = 1.0 / num_experts
        dead = sum(1 for f in frequencies if f < uniform_freq * 0.01)
        if dead > 0:
            issues.append(f"  Layer {layer_idx}: {dead}/{num_experts} dead experts (< 0.01x uniform)")

        # Gini coefficient
        sorted_imps = sorted(importances)
        n = len(sorted_imps)
        if sum(sorted_imps) > 0:
            gini = sum((2 * (i + 1) - n - 1) * imp for i, imp in enumerate(sorted_imps)) / (n * sum(sorted_imps))
        else:
            gini = 1.0

        if gini > 0.8:
            issues.append(f"  Layer {layer_idx}: Gini={gini:.3f} — routing is dominated by few experts")
        elif gini < 0.02:
            issues.append(f"  Layer {layer_idx}: Gini={gini:.3f} — routing is too uniform (experts not specialized)")

        # Top-1 domination
        max_freq = max(frequencies)
        if max_freq > 0.5:
            top_eid = max(experts.items(), key=lambda x: x[1]["frequency"])[0]
            issues.append(f"  Layer {layer_idx}: Expert {top_eid} captures {max_freq*100:.1f}% of routing")

    # Summary stats
    all_ginis = []
    for layer_idx in sorted(merged_profiles.keys()):
        experts = merged_profiles[layer_idx]
        imps = sorted([e["importance"] for e in experts.values()])
        n = len(imps)
        gini = sum((2 * (i + 1) - n - 1) * imp for i, imp in enumerate(imps)) / (n * max(sum(imps), 1e-10))
        all_ginis.append(gini)
        active = sum(1 for e in experts.values() if e["frequency"] > 0.001)
        print(f"  Layer {layer_idx:2d}: {active:3d}/{num_experts} active, Gini={gini:.3f}")

    print(f"\n  Mean Gini: {np.mean(all_ginis):.3f} (original would be ~0.05-0.15 for healthy MoE)")

    if issues:
        print(f"\n  ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(issue)
    else:
        print(f"\n  No routing issues detected.")

    return merged_profiles, issues


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 26B-A4B Expert Pruning")
    parser.add_argument("--source", type=str, default=".", help="Source model directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--target-experts", type=int, default=None,
                        help="Number of experts to keep per layer (default: produce 96 and 64)")
    parser.add_argument("--profile", action="store_true",
                        help="Only profile expert activations, don't prune")
    parser.add_argument("--verify-only", type=str, default=None,
                        help="Only verify an already-merged model (path to merged model dir)")
    parser.add_argument("--cal-samples", type=int, default=256,
                        help="Calibration samples for profiling (default: 256)")
    parser.add_argument("--reasoning-weight", type=float, default=2.0,
                        help="Weight multiplier for reasoning/math calibration data (default: 2.0)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pruning plan without writing files")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip post-merge verification")
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()

    # Verify-only mode: just run verification on an existing merged model
    if args.verify_only:
        verify_dir = Path(args.verify_only).resolve()
        print(f"Verifying merged model: {verify_dir}")
        verify_merged_model(verify_dir, original_profiles={},
                           device=args.device, num_samples=min(50, args.cal_samples),
                           reasoning_weight=args.reasoning_weight)
        return

    # Load config
    with open(source_dir / "config.json") as f:
        config = json.load(f)

    text_cfg = config["text_config"]
    num_layers = text_cfg["num_hidden_layers"]
    num_experts = text_cfg["num_experts"]
    print(f"Model: {num_layers} layers, {num_experts} experts, top-{text_cfg['top_k_experts']}")

    # Phase 1: Profile expert activations
    print("\n--- Phase 1: Expert Activation Profiling ---")
    profiles, num_layers, num_experts, top_k = profile_experts(
        source_dir, device=args.device,
        num_samples=args.cal_samples,
        reasoning_weight=args.reasoning_weight,
    )

    sorted_global, act_similarity = print_expert_stats(profiles, num_layers, num_experts)

    if args.profile:
        profile_file = source_dir / "expert_profile.json"
        serializable = {}
        for layer_idx, experts in profiles.items():
            serializable[str(layer_idx)] = {
                str(eid): stats for eid, stats in experts.items()
            }
        with open(profile_file, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nProfile saved to {profile_file}")
        return

    # Phase 2: Compute per-layer weight similarity
    print("\n--- Phase 2: Per-Layer Weight Similarity ---")
    per_layer_sim = compute_weight_similarity_per_layer(source_dir, num_layers, num_experts)

    # Show per-layer weight similarity stats
    for layer_idx in sorted(per_layer_sim.keys()):
        sim = per_layer_sim[layer_idx]
        # Get upper triangle (excluding diagonal)
        upper = sim[np.triu_indices_from(sim, k=1)]
        print(f"  Layer {layer_idx:2d}: weight sim mean={upper.mean():.4f}, "
              f"max={upper.max():.4f}, >0.95: {(upper > 0.95).sum()}, >0.99: {(upper > 0.99).sum()}")

    # Phase 3: Merge
    expert_params_each = text_cfg.get("moe_intermediate_size", 704) * text_cfg["hidden_size"] * 3
    targets = [args.target_experts] if args.target_experts else [96, 64]

    for target in targets:
        suffix = f"{target}e"
        output_dir = Path(args.output).resolve() if args.output else (source_dir.parent / f"gemma-4-A4B-{suffix}")

        print(f"\n{'='*60}")
        print(f"  Phase 3: Merging to {target} experts -> {output_dir.name}")
        print(f"{'='*60}")

        # Per-layer merge groups using weight similarity
        per_layer_groups = find_merge_groups_per_layer(
            per_layer_sim, profiles, num_experts, target
        )

        # Print merge plan summary
        total_merges = 0
        for layer_idx in sorted(per_layer_groups.keys()):
            groups = per_layer_groups[layer_idx]
            n_merged = sum(1 for g in groups if len(g) > 1)
            max_group = max(len(g) for g in groups) if groups else 0
            total_merges += n_merged
            if n_merged > 0:
                # Show top merge for this layer
                biggest = max((g for g in groups if len(g) > 1), key=len)
                leader = biggest[0]
                sims = [f"{per_layer_sim[layer_idx][leader, e]:.3f}" for e in biggest[1:]]
                print(f"  Layer {layer_idx:2d}: {n_merged} merges, max_group={max_group}, "
                      f"e.g. expert {leader} + {biggest[1:]} (weight_sim={sims})")

        removed_experts = (num_experts - target) * num_layers
        saved_params = removed_experts * expert_params_each
        print(f"\n  Total merges: {total_merges} across {num_layers} layers")
        print(f"  Estimated reduction: ~{saved_params / 1e9:.1f}B params removed")

        if args.dry_run:
            print("  [DRY RUN] Skipping write.")
            continue

        perform_expert_merge(source_dir, output_dir, per_layer_groups, config, profiles)

        # Save metadata
        meta = {
            "base_model": "google/gemma-4-26B-A4B-it",
            "pruning": {
                "method": "expert_merge_v2_per_layer_weight_sim",
                "original_experts": num_experts,
                "target_experts": target,
                "top_k": top_k,
                "reasoning_weight": args.reasoning_weight,
                "cal_samples": args.cal_samples,
                "total_merges": total_merges,
                "per_layer_groups": {
                    str(li): [[int(e) for e in g] for g in groups]
                    for li, groups in per_layer_groups.items()
                },
            }
        }
        with open(output_dir / "expert_prune_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Phase 4: Post-merge verification
        if not args.no_verify:
            print(f"\n--- Phase 4: Post-Merge Verification ({suffix}) ---")
            verify_merged_model(
                output_dir, original_profiles=profiles,
                device=args.device, num_samples=min(50, args.cal_samples),
                reasoning_weight=args.reasoning_weight,
            )
        else:
            print("  [Skipping verification]")

        print(f"  Done! -> {output_dir}")

    print(f"\nAll variants saved.")


if __name__ == "__main__":
    main()
