#!/usr/bin/env python3
"""
Measure actual expert contribution to the residual stream.
Not just "was it activated?" but "how much did it change the hidden state?"

Hooks into the Experts forward to capture:
  contribution = routing_weight * ||expert_output||_2

Usage:
  python expert_contribution.py --source .
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from functools import wraps
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


TOPICS = {
    "math": [
        "Solve step by step: If f(x) = 3x^2 - 2x + 1, find f'(x) and evaluate at x=4.",
        "Calculate the integral of sin(x)*cos(x) dx from 0 to pi/2.",
        "A bag contains 5 red and 3 blue balls. Draw 2 without replacement. P(both red)?",
        "What is 17 * 23? Show your work.",
        "Find all prime numbers p such that p^2 + 2 is also prime.",
        "Solve: 2x^2 + 5x - 3 = 0",
    ],
    "logic": [
        "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
        "You have 8 identical-looking coins, one is heavier. Find it in 2 weighings.",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
        "Three people check into a hotel that costs $30. They each pay $10. The clerk gives $5 back via bellboy who keeps $2, returns $1 each. Each paid $9, total $27 + $2 = $29. Where's the missing dollar?",
        "A farmer has a fox, chicken, grain. Must cross a river carrying one at a time. How?",
        "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
    ],
    "code": [
        "Write a Python function to detect a cycle in a linked list using Floyd's algorithm.",
        "What is the time complexity of quicksort in the average case? Explain why.",
        "Explain why a hash table has O(1) average lookup but O(n) worst case.",
        "Write a function to find the longest common subsequence of two strings.",
        "What's the difference between a mutex and a semaphore?",
        "Implement binary search in Python. Handle edge cases.",
    ],
    "science": [
        "Explain quantum entanglement in simple terms.",
        "How does CRISPR-Cas9 gene editing work?",
        "What is the Higgs boson and why was its discovery important?",
        "Describe the process of nuclear fusion in stars.",
        "What causes antibiotic resistance in bacteria?",
        "Explain how mRNA vaccines work.",
    ],
    "creative": [
        "Write a haiku about the ocean.",
        "Tell me a short story about a robot that learns to paint.",
        "Describe a sunset to someone who has never seen one.",
        "Write a limerick about a cat who loves pizza.",
        "Compose a brief poem about the beauty of mathematics.",
        "Create a metaphor for the passage of time.",
    ],
}


def install_decoder_hooks(model, num_layers, num_experts, tracker):
    """Install hooks on experts that recompute per-expert output norms on GPU in bf16.
    Measures actual contribution: ||routing_weight * expert_output||_2 per expert."""
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

                        # Recompute expert output in bf16 on GPU
                        gate, up = nn.functional.linear(
                            current_state, module.gate_up_proj[expert_idx]
                        ).chunk(2, dim=-1)
                        current_hidden = module.act_fn(gate) * up
                        current_hidden = nn.functional.linear(
                            current_hidden, module.down_proj[expert_idx]
                        )

                        # Raw output norm (before routing weight)
                        raw_norm = current_hidden.float().norm().item()

                        # Weighted output (actual contribution to residual)
                        weights_for_tokens = top_k_weights[token_idx, top_k_pos]
                        weighted = current_hidden * weights_for_tokens.unsqueeze(-1)
                        weighted_norm = weighted.float().norm().item()

                        n_tokens = len(token_idx)
                        avg_weight = weights_for_tokens.float().mean().item()

                        tracker[li][eid]["raw_norm_sum"] += raw_norm
                        tracker[li][eid]["weighted_norm_sum"] += weighted_norm
                        tracker[li][eid]["avg_routing_weight_sum"] += avg_weight
                        tracker[li][eid]["token_count"] += n_tokens
                        tracker[li][eid]["call_count"] += 1

                    # Total MoE output norm
                    total_norm = output.float().norm().item()
                    tracker[li]["__total__"]["weighted_norm_sum"] += total_norm
                    tracker[li]["__total__"]["call_count"] += 1

            return hook

        h = layer.experts.register_forward_hook(make_experts_hook(layer_idx))
        hooks.append(h)

    print(f"  Installed {len(hooks)} expert hooks across {num_layers} layers")
    return hooks


def run_prompts(model, tokenizer, prompts, device, max_new_tokens=256):
    """Run prompts through the model WITH generation (CoT thinking).
    Expert activations during generation are what matter for reasoning."""
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        try:
            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt", return_dict=True,
                add_generation_prompt=True, enable_thinking=True)
            input_ids = inputs["input_ids"].to(device)
        except Exception:
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                                  max_length=512)["input_ids"].to(device)

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


def analyze_tracker(tracker, num_layers, num_experts, topic_name):
    """Analyze contribution data from tracker."""
    result = {}
    for layer_idx in range(num_layers):
        layer_data = tracker[layer_idx]
        total_info = layer_data["__total__"]

        experts = []
        total_wnorm = sum(layer_data[eid]["weighted_norm_sum"] for eid in range(num_experts))
        for eid in range(num_experts):
            d = layer_data[eid]
            if d["call_count"] == 0:
                experts.append({
                    "id": eid, "weighted_norm": 0, "raw_norm": 0,
                    "avg_weight": 0, "tokens": 0, "pct_of_total": 0,
                })
                continue

            experts.append({
                "id": eid,
                "weighted_norm": d["weighted_norm_sum"],
                "raw_norm": d["raw_norm_sum"],
                "avg_weight": d["avg_routing_weight_sum"] / d["call_count"],
                "tokens": d["token_count"],
                "pct_of_total": d["weighted_norm_sum"] / max(total_wnorm, 1e-10) * 100,
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
    parser.add_argument("--source", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()

    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        os.environ["HF_TOKEN"] = open(token_path).read().strip()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import bitsandbytes as bnb
    _orig_new = bnb.nn.Params4bit.__new__
    def _patched_new(cls, *args, **kwargs):
        kwargs.pop("_is_hf_initialized", None)
        return _orig_new(cls, *args, **kwargs)
    bnb.nn.Params4bit.__new__ = _patched_new

    # Load 4-bit fully on GPU — no CPU offload, no meta tensor issues
    print("Loading model 4-bit on GPU...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(source_dir), quantization_config=bnb_config,
        device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(source_dir))
    model.eval()

    text_cfg = model.config.text_config
    num_layers = text_cfg.num_hidden_layers
    num_experts = text_cfg.num_experts
    top_k = text_cfg.top_k_experts
    print(f"  {num_layers} layers, {num_experts} experts, top-{top_k}")

    all_topic_results = {}

    for topic, prompts in TOPICS.items():
        # Use first 3 prompts per topic for speed
        prompts = prompts[:3]
        print(f"\nProfiling: {topic} ({len(prompts)} prompts, 128 gen tokens each)...")

        # Fresh tracker per topic
        tracker = defaultdict(lambda: defaultdict(
            lambda: {"raw_norm_sum": 0.0, "weighted_norm_sum": 0.0,
                     "avg_routing_weight_sum": 0.0, "token_count": 0, "call_count": 0}
        ))

        hooks = install_decoder_hooks(model, num_layers, num_experts, tracker)
        run_prompts(model, tokenizer, prompts, model.device, max_new_tokens=128)
        for h in hooks:
            h.remove()
        result = analyze_tracker(tracker, num_layers, num_experts, topic)
        all_topic_results[topic] = result

    # ── Print analysis ──
    print(f"\n{'='*90}")
    print(f"  EXPERT CONTRIBUTION ANALYSIS (weighted output norm to residual stream)")
    print(f"{'='*90}")

    # Per-topic: how concentrated is the actual contribution?
    print(f"\n  Contribution concentration per topic (weighted output norm):")
    print(f"    {'Topic':>10}  {'Top-1%':>8}  {'Top-8%':>8}  {'Top-16%':>8}  {'Top-32%':>8}  "
          f"{'Gini':>6}  {'80% needs':>9}")
    for topic in TOPICS:
        expert_total = defaultdict(float)
        for layer_idx in range(num_layers):
            for e in all_topic_results[topic][layer_idx]["experts"]:
                expert_total[e["id"]] += e["weighted_norm"]
        total = sum(expert_total.values())
        if total == 0:
            continue

        sorted_contribs = sorted(expert_total.values(), reverse=True)
        cumsum = np.cumsum(sorted_contribs) / total * 100

        top1_pct = sorted_contribs[0] / total * 100
        top8_pct = sum(sorted_contribs[:8]) / total * 100
        top16_pct = sum(sorted_contribs[:16]) / total * 100
        top32_pct = sum(sorted_contribs[:32]) / total * 100

        needs_80 = int(np.searchsorted(cumsum, 80)) + 1

        n = len(sorted_contribs)
        sorted_asc = sorted(sorted_contribs)
        gini = sum((2*(i+1) - n - 1) * v for i, v in enumerate(sorted_asc)) / (n * total)

        print(f"    {topic:>10}  {top1_pct:>7.1f}%  {top8_pct:>7.1f}%  {top16_pct:>7.1f}%  "
              f"{top32_pct:>7.1f}%  {gini:>6.3f}  {needs_80:>5}")

    # Per-layer distribution (math)
    print(f"\n  Per-layer contribution distribution (math topic):")
    print(f"    {'Layer':>5}  {'MoE norm':>10}  {'Top-1 expert':>12}  {'Top-1 %':>7}  "
          f"{'Top-8 %':>7}  {'Active':>6}  {'Bottom-64 %':>11}")
    for layer_idx in range(num_layers):
        data = all_topic_results["math"][layer_idx]
        total_w = data["total_expert_wnorm"]
        if total_w == 0:
            print(f"    {layer_idx:>5}  {'(no data)':>10}")
            continue
        experts = data["experts"]
        top1 = experts[0]
        top8_w = sum(e["weighted_norm"] for e in experts[:8])
        bottom64_w = sum(e["weighted_norm"] for e in experts[64:])
        active = sum(1 for e in experts if e["tokens"] > 0)

        print(f"    {layer_idx:>5}  {data['total_moe_norm']:>10.1f}  e{top1['id']:>3d}({top1['avg_weight']:.3f})  "
              f"{top1['pct_of_total']:>6.1f}%  {top8_w/total_w*100:>6.1f}%  "
              f"{active:>6}  {bottom64_w/total_w*100:>10.1f}%")

    # Compare math vs creative top contributors
    print(f"\n  Top-10 contributing experts: math vs creative (by weighted output norm)")
    math_total = defaultdict(float)
    creative_total = defaultdict(float)
    for li in range(num_layers):
        for e in all_topic_results["math"][li]["experts"]:
            math_total[e["id"]] += e["weighted_norm"]
        for e in all_topic_results["creative"][li]["experts"]:
            creative_total[e["id"]] += e["weighted_norm"]

    math_sorted = sorted(math_total.items(), key=lambda x: x[1], reverse=True)
    creative_sorted = sorted(creative_total.items(), key=lambda x: x[1], reverse=True)
    m_total = sum(v for _, v in math_sorted)
    c_total = sum(v for _, v in creative_sorted)

    print(f"    {'Rank':>4}  {'Math expert':>11} {'Math %':>7}  {'Creative expert':>15} {'Creative %':>10}")
    for rank in range(10):
        me, mv = math_sorted[rank]
        ce, cv = creative_sorted[rank]
        print(f"    #{rank+1:>3}  e{me:>3d}       {mv/m_total*100:>6.1f}%  "
              f"e{ce:>3d}             {cv/c_total*100:>6.1f}%")

    # Overlap analysis
    math_top32 = set(eid for eid, _ in math_sorted[:32])
    creative_top32 = set(eid for eid, _ in creative_sorted[:32])
    overlap = len(math_top32 & creative_top32)
    print(f"\n  Math vs Creative top-32 overlap: {overlap}/32 shared")

    # Per-layer overlap of top-8
    print(f"\n  Per-layer top-8 overlap (math vs creative):")
    for li in range(num_layers):
        m_experts = {e["id"]: e["weighted_norm"] for e in all_topic_results["math"][li]["experts"]}
        c_experts = {e["id"]: e["weighted_norm"] for e in all_topic_results["creative"][li]["experts"]}
        m_top8 = set(sorted(m_experts, key=m_experts.get, reverse=True)[:8])
        c_top8 = set(sorted(c_experts, key=c_experts.get, reverse=True)[:8])
        ovl = len(m_top8 & c_top8)
        m_only = m_top8 - c_top8
        c_only = c_top8 - m_top8
        print(f"    L{li:2d}: {ovl}/8 shared, math-only={sorted(m_only)}, creative-only={sorted(c_only)}")

    # Save full results
    os.makedirs("eval_results", exist_ok=True)
    # Convert to serializable
    save_data = {}
    for topic, result in all_topic_results.items():
        save_data[topic] = {}
        for li, data in result.items():
            save_data[topic][str(li)] = {
                "total_norm": data["total_norm"],
                "experts": data["experts"][:20],  # top 20 only to keep file small
            }
    with open("eval_results/expert_contributions.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to eval_results/expert_contributions.json")


if __name__ == "__main__":
    main()
