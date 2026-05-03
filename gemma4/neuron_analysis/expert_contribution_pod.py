#!/usr/bin/env python3
"""Pod-friendly variant of expert_contribution.py — no bnb 4-bit.

bnb 4-bit does NOT quantize MoE stacked parameters on Gemma 4, so experts stay
in bf16 (~44 GB) — won't fit 16 GB VRAM on the pod's 2× RTX 3070s.

This script loads the model in pure bf16 with `device_map="auto"` and
`max_memory` split: dense backbone → GPU (fast), experts → CPU (slow but works).

Output: writes a JSON file with per-topic per-layer total_norm and per-expert
weighted contribution, same format as the original expert_contribution.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


# Same prompt set as original (trimmed to 3 per topic for speed)
TOPICS = {
    "math": [
        "Solve step by step: If f(x) = 3x^2 - 2x + 1, find f'(x) and evaluate at x=4.",
        "Calculate the integral of sin(x)*cos(x) dx from 0 to pi/2.",
        "A bag contains 5 red and 3 blue balls. Draw 2 without replacement. P(both red)?",
    ],
    "logic": [
        "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
        "You have 8 identical-looking coins, one is heavier. Find it in 2 weighings.",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
    ],
    "code": [
        "Write a Python function to detect a cycle in a linked list using Floyd's algorithm.",
        "What is the time complexity of quicksort in the average case? Explain why.",
        "Explain why a hash table has O(1) average lookup but O(n) worst case.",
    ],
    "science": [
        "Explain quantum entanglement in simple terms.",
        "How does CRISPR-Cas9 gene editing work?",
        "What is the Higgs boson and why was its discovery important?",
    ],
    "creative": [
        "Write a haiku about the ocean.",
        "Tell me a short story about a robot that learns to paint.",
        "Describe a sunset to someone who has never seen one.",
    ],
}


def install_decoder_hooks(model, num_layers, num_experts, tracker):
    """Install forward hooks on experts. Uses .float().norm() to avoid bf16 overflow."""
    layers = model.model.language_model.layers
    hooks = []

    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        if not hasattr(layer, "experts"):
            continue

        def make_experts_hook(li):
            def hook(module, args, output):
                hidden_states, top_k_index, top_k_weights = args
                n_experts = module.num_experts

                with torch.no_grad():
                    expert_mask = torch.nn.functional.one_hot(
                        top_k_index, num_classes=n_experts
                    ).permute(2, 1, 0)
                    expert_hit = torch.greater(
                        expert_mask.sum(dim=(-1, -2)), 0
                    ).nonzero()

                    for expert_idx in expert_hit:
                        expert_idx = expert_idx[0]
                        if expert_idx == n_experts:
                            continue
                        eid = int(expert_idx)
                        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                        current_state = hidden_states[token_idx]

                        gate, up = nn.functional.linear(
                            current_state, module.gate_up_proj[expert_idx]
                        ).chunk(2, dim=-1)
                        current_hidden = module.act_fn(gate) * up
                        current_hidden = nn.functional.linear(
                            current_hidden, module.down_proj[expert_idx]
                        )

                        # fp32 reductions — avoid bf16 overflow in layers 11-29
                        raw_norm = current_hidden.float().norm().item()

                        weights_for_tokens = top_k_weights[token_idx, top_k_pos]
                        weighted = current_hidden * weights_for_tokens.unsqueeze(-1)
                        weighted_norm = weighted.float().norm().item()

                        import math
                        if math.isnan(weighted_norm) or math.isinf(weighted_norm):
                            continue

                        n_tokens = len(token_idx)
                        avg_weight = weights_for_tokens.float().mean().item()

                        tracker[li][eid]["raw_norm_sum"] += raw_norm
                        tracker[li][eid]["weighted_norm_sum"] += weighted_norm
                        tracker[li][eid]["avg_routing_weight_sum"] += avg_weight
                        tracker[li][eid]["token_count"] += n_tokens
                        tracker[li][eid]["call_count"] += 1

                    total_norm = output.float().norm().item()
                    tracker[li]["__total__"]["weighted_norm_sum"] += total_norm
                    tracker[li]["__total__"]["call_count"] += 1

            return hook

        h = layer.experts.register_forward_hook(make_experts_hook(layer_idx))
        hooks.append(h)

    print(f"  Installed {len(hooks)} expert hooks across {num_layers} layers")
    return hooks


def run_prompts(model, tokenizer, prompts, max_new_tokens=64):
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        try:
            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt", return_dict=True,
                add_generation_prompt=True)
            input_ids = inputs["input_ids"].to(model.device)
        except Exception:
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                                  max_length=512)["input_ids"].to(model.device)

        with torch.no_grad():
            try:
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                gen_len = out.shape[1] - input_ids.shape[1]
                print(f"    [{i+1}/{len(prompts)}] {gen_len} tokens generated")
            except Exception as e:
                print(f"    [{i+1}/{len(prompts)}] Error: {e}")
                continue


def analyze_tracker(tracker, num_layers, num_experts):
    result = {}
    for layer_idx in range(num_layers):
        layer_data = tracker[layer_idx]
        total_info = layer_data["__total__"]

        experts = []
        total_wnorm = sum(layer_data[eid]["weighted_norm_sum"] for eid in range(num_experts))
        for eid in range(num_experts):
            d = layer_data[eid]
            experts.append({
                "id": eid,
                "weighted_norm": d["weighted_norm_sum"],
                "raw_norm": d["raw_norm_sum"],
                "avg_weight": (d["avg_routing_weight_sum"] / max(d["call_count"], 1)),
                "tokens": d["token_count"],
            })

        experts.sort(key=lambda x: x["weighted_norm"], reverse=True)
        result[layer_idx] = {
            "total_moe_norm": total_info.get("weighted_norm_sum", 0),
            "total_expert_wnorm": total_wnorm,
            "experts": experts,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="HF model dir")
    parser.add_argument("--output", type=str, default="expert_contributions.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gpu-mem", type=str, default="7GiB",
                        help="Per-GPU max memory (pod has 2x8GB 3070s)")
    parser.add_argument("--cpu-mem", type=str, default="200GiB")
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()

    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        os.environ["HF_TOKEN"] = open(token_path).read().strip()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {source_dir.name} in bf16 with GPU+CPU split...")
    print(f"  GPU 0: {args.gpu_mem}, GPU 1: {args.gpu_mem}, CPU: {args.cpu_mem}")

    n_gpus = torch.cuda.device_count()
    max_memory = {i: args.gpu_mem for i in range(n_gpus)}
    max_memory["cpu"] = args.cpu_mem

    model = AutoModelForCausalLM.from_pretrained(
        str(source_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(source_dir))
    model.eval()

    text_cfg = model.config.text_config
    num_layers = text_cfg.num_hidden_layers
    num_experts = text_cfg.num_experts
    print(f"  {num_layers} layers, {num_experts} experts")

    all_topic_results = {}
    for topic, prompts in TOPICS.items():
        print(f"\n=== Topic: {topic} ({len(prompts)} prompts) ===")
        tracker = defaultdict(lambda: defaultdict(
            lambda: {"raw_norm_sum": 0.0, "weighted_norm_sum": 0.0,
                     "avg_routing_weight_sum": 0.0, "token_count": 0, "call_count": 0}
        ))
        hooks = install_decoder_hooks(model, num_layers, num_experts, tracker)
        run_prompts(model, tokenizer, prompts, max_new_tokens=args.max_new_tokens)
        for h in hooks:
            h.remove()
        all_topic_results[topic] = analyze_tracker(tracker, num_layers, num_experts)
        gc.collect()
        torch.cuda.empty_cache()

    # Serializable output
    save_data = {}
    for topic, result in all_topic_results.items():
        save_data[topic] = {}
        for li, data in result.items():
            save_data[topic][str(li)] = {
                "total_moe_norm": data["total_moe_norm"],
                "total_expert_wnorm": data["total_expert_wnorm"],
                "experts_top20": data["experts"][:20],
            }

    with open(args.output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Summary
    print(f"\n=== Per-layer total_moe_norm (mean across topics) ===")
    layer_scores = defaultdict(list)
    for topic in TOPICS:
        for li in range(num_layers):
            layer_scores[li].append(save_data[topic][str(li)]["total_moe_norm"])
    layer_mean = {li: np.mean(scores) for li, scores in layer_scores.items()}
    ranked = sorted(layer_mean.items(), key=lambda x: x[1], reverse=True)
    print(f"{'rank':>4}  {'layer':>5}  {'mean_moe_norm':>15}")
    for rank, (li, score) in enumerate(ranked):
        print(f"  {rank+1:>3}  L{li:<4}  {score:>15.4e}")


if __name__ == "__main__":
    main()
