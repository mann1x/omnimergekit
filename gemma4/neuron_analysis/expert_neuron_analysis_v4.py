#!/usr/bin/env python3
"""
Expert + neuron analysis v4 — same prompts and budget as the working OLD analysis,
but with per-neuron tracking added.

Strategy: replicate the methodology that produced a working 109e drop (5 domains x
8 simple prompts at 128 token budget) and just add the per-neuron data we need
for the residual expert.

Output: scripts/expert_neuron_v4.json
"""

import os
import time
import json
import torch
import numpy as np
from collections import defaultdict
from torch import nn

os.environ["HF_TOKEN"] = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# OLD prompts (from expert_contribution_v2.py — known to produce a working 109e)
PROMPTS = {
    "math": [
        "Solve step by step: If f(x) = 3x^2 - 2x + 1, find f'(x) and evaluate at x=4.",
        "Calculate the integral of sin(x)*cos(x) dx from 0 to pi/2.",
        "What is 17 * 23? Show your work.",
        "A bag contains 5 red and 3 blue balls. Draw 2 without replacement. P(both red)?",
        "Find all prime numbers p such that p^2 + 2 is also prime.",
        "Solve: 2x^2 + 5x - 3 = 0",
        "If a matrix A = [[1,2],[3,4]], find A^(-1) and verify AA^(-1) = I.",
        "Solve the differential equation dy/dx = y*sin(x), y(0) = 1.",
    ],
    "logic": [
        "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
        "You have 8 identical coins, one heavier. Find it in 2 weighings.",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
        "Three people check into a hotel that costs $30. They each pay $10. The clerk gives $5 back via bellboy who keeps $2, returns $1 each. Each paid $9, total $27 + $2 = $29. Where's the missing dollar?",
        "A farmer has a fox, chicken, grain. Must cross a river carrying one at a time. How?",
        "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
        "In a room of 23 people, what's the probability at least two share a birthday?",
        "You see a bear. You walk 1 mile south, 1 mile east, 1 mile north, and you're back where you started. What color is the bear?",
    ],
    "code": [
        "Write a Python function to detect a cycle in a linked list using Floyd's algorithm.",
        "What is the time complexity of quicksort in the average case? Explain why.",
        "Implement binary search in Python. Handle edge cases.",
        "Write a function to find the longest common subsequence of two strings.",
        "What's the difference between a mutex and a semaphore?",
        "Explain why a hash table has O(1) average lookup but O(n) worst case.",
        "Write a Python function to serialize and deserialize a binary tree.",
        "What happens when you recursively compute fibonacci(50) without memoization?",
    ],
    "science": [
        "Explain quantum entanglement in simple terms.",
        "How does CRISPR-Cas9 gene editing work?",
        "Describe the process of nuclear fusion in stars.",
        "What is the Higgs boson and why was its discovery important?",
        "What causes antibiotic resistance in bacteria?",
        "Explain how mRNA vaccines work.",
        "What is dark matter and how do we know it exists?",
        "Describe how plate tectonics shape the Earth's surface.",
    ],
    "creative": [
        "Write a haiku about the ocean.",
        "Tell me a short story about a robot that learns to paint.",
        "Describe a sunset to someone who has never seen one.",
        "Write a limerick about a cat who loves pizza.",
        "Compose a brief poem about the beauty of mathematics.",
        "Create a metaphor for the passage of time.",
        "Write a short dialogue between the Moon and the Sun.",
        "Describe the taste of music to someone who has never heard a song.",
    ],
}

MODEL_PATH = "google/gemma-4-26B-A4B-it"
MAX_NEW_TOKENS = 128  # SAME as old working analysis
OUTPUT_FILE = "scripts/expert_neuron_v4.json"


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model from {MODEL_PATH} (fp16, CPU)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    print(f"Loaded in {time.time()-t0:.0f}s")
    return model, tokenizer


def phase1_generate(model, tokenizer, prompt, max_new_tokens=128):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True,
        add_generation_prompt=True, enable_thinking=True)
    input_ids = inputs["input_ids"]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0], input_ids.shape[1], out.shape[1] - input_ids.shape[1], time.time() - t0


def phase2_profile(model, full_sequence, num_layers, num_experts, intermediate_size):
    """Replay full sequence with hooks. Captures per-expert + per-neuron data."""
    tracker = defaultdict(lambda: defaultdict(lambda: {
        "wnorm": 0.0, "rnorm": 0.0, "wsum": 0.0, "tc": 0, "cc": 0,
        "neuron_act": torch.zeros(intermediate_size),
    }))
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
                    mask = nn.functional.one_hot(top_k_idx, num_classes=n).permute(2, 1, 0)
                    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
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
                        # Per-neuron solicitation
                        neuron_sq = (intermediate.float() ** 2).sum(dim=0)
                        tracker[layer_idx][eid]["neuron_act"] += neuron_sq.cpu()
                        # Per-expert importance
                        eh = nn.functional.linear(intermediate, module.down_proj[eid])
                        wt = top_k_wt[tidx, pos]
                        weighted = eh * wt.unsqueeze(-1)
                        tracker[layer_idx][eid]["wnorm"] += weighted.norm().item()
                        tracker[layer_idx][eid]["rnorm"] += eh.norm().item()
                        tracker[layer_idx][eid]["wsum"] += wt.float().mean().item()
                        tracker[layer_idx][eid]["tc"] += len(tidx)
                        tracker[layer_idx][eid]["cc"] += 1
            return hook
        hooks.append(layer.experts.register_forward_hook(make_hook(li)))

    input_ids = full_sequence.unsqueeze(0)
    mm_ids = torch.zeros_like(input_ids)
    t0 = time.time()
    with torch.no_grad():
        model(input_ids, mm_token_type_ids=mm_ids)
    elapsed = time.time() - t0
    for h in hooks:
        h.remove()
    return tracker, elapsed


def merge_trackers(target, source, num_layers, num_experts):
    for li in range(num_layers):
        for eid in range(num_experts):
            for key in ["wnorm", "rnorm", "wsum", "tc", "cc"]:
                target[li][eid][key] += source[li][eid][key]
            target[li][eid]["neuron_act"] += source[li][eid]["neuron_act"]


def main():
    print(f"=== Expert + Neuron Analysis v4 (CPU fp16, OLD prompts + 128 tok) ===")
    total_prompts = sum(len(v) for v in PROMPTS.values())
    print(f"Total: {total_prompts} prompts in {len(PROMPTS)} categories")

    model, tokenizer = load_model()
    num_layers = model.config.text_config.num_hidden_layers
    num_experts = model.config.text_config.num_experts
    intermediate_size = model.config.text_config.moe_intermediate_size
    hidden_size = model.config.text_config.hidden_size
    print(f"Layers: {num_layers}, Experts: {num_experts}, Intermediate: {intermediate_size}\n")

    all_results = {}
    overall_start = time.time()
    prompt_idx = 0

    for category, prompts in PROMPTS.items():
        print(f"\n=== {category} ({len(prompts)}) ===")
        cat_tracker = defaultdict(lambda: defaultdict(lambda: {
            "wnorm": 0.0, "rnorm": 0.0, "wsum": 0.0, "tc": 0, "cc": 0,
            "neuron_act": torch.zeros(intermediate_size),
        }))

        for i, prompt in enumerate(prompts):
            prompt_idx += 1
            full_seq, plen, glen, gen_t = phase1_generate(
                model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
            tracker, prof_t = phase2_profile(
                model, full_seq, num_layers, num_experts, intermediate_size)
            merge_trackers(cat_tracker, tracker, num_layers, num_experts)
            elapsed = (time.time() - overall_start) / 60
            eta = elapsed / prompt_idx * (total_prompts - prompt_idx)
            print(f"  [{prompt_idx}/{total_prompts}] {category} #{i+1}: "
                  f"{glen}t {gen_t:.0f}s + prof {prof_t:.0f}s "
                  f"(elapsed {elapsed:.0f}m ETA {eta:.0f}m)")

        all_results[category] = cat_tracker

    print(f"\nSaving to {OUTPUT_FILE}...")
    save_data = {
        "metadata": {
            "model": MODEL_PATH,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "intermediate_size": intermediate_size,
            "hidden_size": hidden_size,
            "max_new_tokens": MAX_NEW_TOKENS,
            "categories": {k: len(v) for k, v in PROMPTS.items()},
            "note": "Same prompts/budget as working v2 OLD analysis, with per-neuron data added",
        },
        "categories": {},
    }
    for category, tracker in all_results.items():
        cat_data = {}
        for li in range(num_layers):
            layer_data = []
            for eid in range(num_experts):
                d = tracker[li][eid]
                layer_data.append({
                    "id": eid,
                    "wnorm": d["wnorm"],
                    "rnorm": d["rnorm"],
                    "wsum": d["wsum"],
                    "tc": d["tc"],
                    "cc": d["cc"],
                    "neuron_act": d["neuron_act"].tolist(),
                })
            cat_data[str(li)] = layer_data
        save_data["categories"][category] = cat_data

    with open(OUTPUT_FILE, "w") as f:
        json.dump(save_data, f)

    total_min = (time.time() - overall_start) / 60
    print(f"\nDone in {total_min:.0f} min. Saved to {OUTPUT_FILE}")
    print(f"Size: {os.path.getsize(OUTPUT_FILE)/1024**2:.0f} MB")


if __name__ == "__main__":
    main()
