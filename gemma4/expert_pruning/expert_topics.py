#!/usr/bin/env python3
"""
Analyze which experts activate for different topics/domains.
Shows per-topic expert activation heatmap to understand specialization.

Usage:
  python expert_topics.py --source .
"""

from __future__ import annotations

import argparse
import gc
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
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
        "Three people check into a hotel room that costs $30. They split it $10 each. The clerk realizes the room is $25 and gives $5 to the bellboy. The bellboy keeps $2 and returns $1 each. Now each paid $9, total $27, plus $2 = $29. Where's the missing dollar?",
        "A farmer has a fox, chicken, and grain. He must cross a river in a boat that carries only one item besides himself. How?",
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
    "history": [
        "What caused the fall of the Roman Empire?",
        "Explain the key events of the French Revolution.",
        "What was the significance of the Magna Carta?",
        "Describe the causes and effects of World War I.",
        "What was the Silk Road and why was it important?",
        "How did the Industrial Revolution transform society?",
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


def profile_topic(model, tokenizer, prompts, device, num_layers, num_experts):
    """Run prompts through model and capture per-layer expert activation."""
    # layer -> expert_id -> {count, weight_sum, tokens}
    layer_stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "weight_sum": 0.0}))
    total_tokens = 0
    hooks = []

    def make_hook(layer_idx):
        def hook(module, args, output):
            nonlocal total_tokens
            if not isinstance(output, tuple) or len(output) < 3:
                return
            top_k_weights = output[1].detach().float().cpu()  # [B*S, K]
            top_k_index = output[2].detach().cpu()            # [B*S, K]

            for k in range(top_k_index.shape[1]):
                for token_pos in range(top_k_index.shape[0]):
                    eid = int(top_k_index[token_pos, k])
                    ew = float(top_k_weights[token_pos, k])
                    layer_stats[layer_idx][eid]["count"] += 1
                    layer_stats[layer_idx][eid]["weight_sum"] += ew

            if layer_idx == 0:
                total_tokens += top_k_index.shape[0]
        return hook

    layers = model.model.language_model.layers
    for i in range(num_layers):
        if hasattr(layers[i], "router"):
            hooks.append(layers[i].router.register_forward_hook(make_hook(i)))

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)["input_ids"]
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"]

        inputs = inputs.to(device)
        mm_ids = torch.zeros_like(inputs)
        with torch.no_grad():
            try:
                model(inputs, mm_token_type_ids=mm_ids)
            except Exception:
                continue

    for h in hooks:
        h.remove()

    return dict(layer_stats), total_tokens


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

    print("Loading model in 4-bit...")
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
    print(f"  {num_layers} layers, {num_experts} experts, top-{top_k}")

    # Profile each topic
    topic_profiles = {}
    for topic, prompts in TOPICS.items():
        print(f"\nProfiling: {topic} ({len(prompts)} prompts)...")
        stats, n_tokens = profile_topic(model, tokenizer, prompts, model.device,
                                         num_layers, num_experts)
        topic_profiles[topic] = (stats, n_tokens)
        print(f"  {n_tokens} tokens processed")

    # Analysis
    print(f"\n{'='*80}")
    print(f"  EXPERT TOPIC SPECIALIZATION ANALYSIS")
    print(f"{'='*80}")

    # For each topic, compute which experts are most activated (globally across layers)
    topic_expert_importance = {}  # topic -> expert_id -> importance
    for topic, (stats, n_tokens) in topic_profiles.items():
        expert_imp = defaultdict(float)
        for layer_idx, experts in stats.items():
            for eid, s in experts.items():
                # Normalize by tokens to make comparable across topics
                expert_imp[eid] += s["weight_sum"] / max(n_tokens, 1)
        topic_expert_importance[topic] = dict(expert_imp)

    # Find topic-specific experts: high activation for one topic, low for others
    print(f"\n  Top 10 experts per topic (by routing weight share):")
    for topic in TOPICS:
        imp = topic_expert_importance[topic]
        sorted_experts = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        top10 = sorted_experts[:10]
        print(f"\n  {topic.upper()}:")
        for rank, (eid, importance) in enumerate(top10):
            # Check if this expert is specific to this topic
            other_imps = [topic_expert_importance[t].get(eid, 0) for t in TOPICS if t != topic]
            avg_other = np.mean(other_imps) if other_imps else 0
            specificity = importance / max(avg_other, 1e-10)
            print(f"    #{rank+1:2d}  Expert {eid:3d}  imp={importance:.5f}  "
                  f"specificity={specificity:.2f}x vs other topics")

    # How many unique experts does each topic use significantly?
    print(f"\n  Expert utilization per topic (experts with >1% of max importance):")
    for topic in TOPICS:
        imp = topic_expert_importance[topic]
        if not imp:
            continue
        max_imp = max(imp.values())
        threshold = max_imp * 0.01
        active = sum(1 for v in imp.values() if v > threshold)
        top8_share = sum(sorted(imp.values(), reverse=True)[:8]) / max(sum(imp.values()), 1e-10) * 100
        print(f"    {topic:>10}: {active:3d}/{num_experts} experts active, "
              f"top-8 share: {top8_share:.1f}%")

    # Per-layer analysis for math vs creative (most different expected)
    print(f"\n  Per-layer: math vs creative expert overlap:")
    print(f"    {'Layer':>5}  {'Math top-8':>30}  {'Creative top-8':>30}  {'Overlap':>7}")
    for layer_idx in range(num_layers):
        math_experts = topic_profiles["math"][0].get(layer_idx, {})
        creative_experts = topic_profiles["creative"][0].get(layer_idx, {})

        math_top8 = sorted(math_experts.items(), key=lambda x: x[1]["weight_sum"], reverse=True)[:8]
        creative_top8 = sorted(creative_experts.items(), key=lambda x: x[1]["weight_sum"], reverse=True)[:8]

        math_ids = set(eid for eid, _ in math_top8)
        creative_ids = set(eid for eid, _ in creative_top8)
        overlap = len(math_ids & creative_ids)

        print(f"    {layer_idx:>5}  {str(sorted(math_ids)):>30}  "
              f"{str(sorted(creative_ids)):>30}  {overlap:>3}/8")

    # Find experts that are exclusive to one topic
    print(f"\n  Topic-exclusive experts (>3x specificity, appear in top-20 for topic):")
    for topic in TOPICS:
        imp = topic_expert_importance[topic]
        sorted_e = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]
        exclusives = []
        for eid, importance in sorted_e:
            other_imps = [topic_expert_importance[t].get(eid, 0) for t in TOPICS if t != topic]
            avg_other = np.mean(other_imps) if other_imps else 0
            if avg_other > 0 and importance / avg_other > 3.0:
                exclusives.append((eid, importance, importance / avg_other))
        if exclusives:
            ex_str = ", ".join(f"e{eid}({spec:.1f}x)" for eid, _, spec in exclusives[:5])
            print(f"    {topic:>10}: {ex_str}")
        else:
            print(f"    {topic:>10}: none found (experts are shared)")


if __name__ == "__main__":
    main()
