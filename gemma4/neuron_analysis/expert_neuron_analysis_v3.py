#!/usr/bin/env python3
"""
Expert + neuron analysis v3 — improved methodology.

Key improvements over v2:
- Mid-difficulty prompts that the model can FINISH in 2048 tokens
- Filters incomplete prompts (those hitting max_tokens) from the accumulator
- Different prompts than the old simple analysis (designed to ADD new signal)
- Tracks completion stats per category
- Same per-expert + per-neuron data structure as v2

Total: 5 domains x 8 prompts + 9 GPQA hard = 49 prompts.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from collections import defaultdict
from torch import nn

os.environ["HF_TOKEN"] = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Mid-difficulty prompts: 2-4 step reasoning, expected 300-1500 tokens to answer.
# Designed to be DIFFERENT from old simple prompts (add new signal).
PROMPTS = {
    "math": [
        # Word problems requiring 2-4 steps
        "A car accelerates uniformly from 10 m/s to 30 m/s over 5 seconds. Find the acceleration and the distance traveled.",
        "If sin(x) = 3/5 and x is in the second quadrant, find cos(x), tan(x), and sin(2x).",
        "Find the volume of a cone with height 12 and base radius 5. Then find its lateral surface area.",
        "A geometric series has first term 3 and common ratio 0.4. Find the sum of the first 6 terms and the infinite sum.",
        "Solve the system: 2x + 3y = 13, 5x - y = 8. Verify your answer.",
        "Find the derivative of f(x) = x*ln(x) - x using the product rule. Then find its critical points.",
        "What is the probability of rolling exactly two 6s in five rolls of a fair die? Use the binomial formula.",
        "Find the equation of the tangent line to y = x^3 - 2x + 1 at the point where x = 2.",
    ],
    "logic": [
        "If today is Wednesday, what day of the week will it be in 100 days? Show the modular arithmetic.",
        "Three friends Alice, Bob, Carol have ages summing to 60. Alice is twice Bob's age. Carol is 5 years younger than Alice. Find each age.",
        "A boat travels downstream 24 km in 2 hours and upstream 18 km in 3 hours. Find the boat speed and current speed.",
        "If A implies B, and B implies C, and C implies not A, what can you conclude about A? Explain with truth values.",
        "Five houses in a row, each a different color. The red house is between the blue and green. The yellow is at one end. The white is next to the red. List a valid order.",
        "If a clock shows 3:15, what is the angle between the hour and minute hands? Show the calculation.",
        "A snail climbs 3 feet up a 10-foot wall during the day but slips 2 feet at night. How many days until it reaches the top? Explain.",
        "You have a 5-liter and a 3-liter jug, and a water source. How do you measure exactly 4 liters? Step by step.",
    ],
    "code": [
        "Write a Python function that returns the nth Fibonacci number using dynamic programming. Explain why this is O(n).",
        "Write a Python function to check if a string is a palindrome, ignoring case and non-alphanumeric characters.",
        "Implement bubble sort in Python with a single optimization: stop early if no swaps occur in a pass.",
        "Write a Python function to merge two sorted lists into a single sorted list. Don't use built-in sort.",
        "Write a Python function to count the frequency of words in a string and return the top 3 most common.",
        "Implement a stack class in Python with push, pop, peek, and is_empty methods. Use a list internally.",
        "Write a Python function to check if a binary tree is a valid binary search tree. Use recursion.",
        "Write a Python function to find the maximum subarray sum using Kadane's algorithm. Explain the invariant.",
    ],
    "science": [
        "Explain how a refrigerator works using the principles of thermodynamics. Include the role of the refrigerant cycle.",
        "Why is the sky blue during the day but red at sunset? Explain Rayleigh scattering.",
        "How do vaccines train the immune system? Describe the role of B cells and memory cells.",
        "Explain how a transistor works as both a switch and an amplifier in a simple circuit.",
        "What causes ocean tides? Describe the Moon's gravitational role and why there are two high tides per day.",
        "Explain photosynthesis: the light-dependent and light-independent reactions, and the net products.",
        "How does GPS determine your location? Describe the role of multiple satellites and time triangulation.",
        "Explain what causes a rainbow: refraction, dispersion, and total internal reflection in water droplets.",
    ],
    "creative": [
        "Write a six-line poem about a programmer who discovers their first bug, using the rhyme scheme ABABCC.",
        "Compose a 100-word short story about a clockmaker who fixes time itself.",
        "Describe a city in three sentences, evoking smell, sound, and a single visual detail.",
        "Write a brief letter from a future archaeologist describing the discovery of a 21st-century smartphone.",
        "Invent a proverb about the relationship between knowledge and humility, then explain its meaning in 2-3 sentences.",
        "Write a haiku and a tanka about the same subject (autumn leaves), then compare how the forms differ.",
        "Describe what music would taste like, what a color would sound like, and what a memory would smell like.",
        "Write a 50-word origin story for a mythical creature that protects libraries.",
    ],
}

GPQA_HARD = {
    "Biology": [96, 43, 101],
    "Chemistry": [63, 125, 21],
    "Physics": [83, 163, 176],
}

MODEL_PATH = "google/gemma-4-26B-A4B-it"
MAX_NEW_TOKENS = 2048
OUTPUT_FILE = "scripts/expert_neuron_v3.json"


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model from {MODEL_PATH} (bf16, CPU)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    print(f"Loaded in {time.time()-t0:.0f}s")
    return model, tokenizer


def load_gpqa_prompts():
    from datasets import load_dataset
    ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')['train']
    gpqa_prompts = {}
    for domain, doc_ids in GPQA_HARD.items():
        gpqa_prompts[f"gpqa_{domain.lower()}"] = []
        for did in doc_ids:
            ex = ds[did]
            q = ex['Question']
            choices = [ex['Correct Answer'], ex['Incorrect Answer 1'],
                       ex['Incorrect Answer 2'], ex['Incorrect Answer 3']]
            import random
            random.seed(did)
            random.shuffle(choices)
            labels = ['A', 'B', 'C', 'D']
            choices_text = '\n'.join(f'({l}) {c}' for l, c in zip(labels, choices))
            prompt = (f"What is the correct answer to this question: {q}\n\n"
                      f"Choices:\n{choices_text}\n\n"
                      f'Format your response as follows: '
                      f'"The correct answer is (insert answer here)".')
            gpqa_prompts[f"gpqa_{domain.lower()}"].append(prompt)
    return gpqa_prompts


def phase1_generate(model, tokenizer, prompt, max_new_tokens=2048):
    """Generate with EOS-aware stopping. Returns (sequence, prompt_len, gen_len, hit_max, elapsed)."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True,
        add_generation_prompt=True, enable_thinking=True)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - t0
    gen_len = out.shape[1] - prompt_len
    hit_max = (gen_len >= max_new_tokens)
    return out[0], prompt_len, gen_len, hit_max, elapsed


def phase2_profile(model, full_sequence, num_layers, num_experts, intermediate_size):
    """Single forward pass with hooks for per-expert + per-neuron stats."""
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

                        neuron_sq = (intermediate.float() ** 2).sum(dim=0)
                        tracker[layer_idx][eid]["neuron_act"] += neuron_sq.cpu()

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
    print("=== Expert + Neuron Analysis v3 (CPU bf16, EOS-aware) ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Max tokens per prompt: {MAX_NEW_TOKENS}")

    gpqa_prompts = load_gpqa_prompts()
    all_prompts = {**PROMPTS, **gpqa_prompts}
    total_prompts = sum(len(v) for v in all_prompts.values())
    print(f"Total: {total_prompts} prompts in {len(all_prompts)} categories")

    model, tokenizer = load_model()
    num_layers = model.config.text_config.num_hidden_layers
    num_experts = model.config.text_config.num_experts
    intermediate_size = model.config.text_config.moe_intermediate_size
    hidden_size = model.config.text_config.hidden_size
    print(f"Layers: {num_layers}, Experts: {num_experts}, "
          f"Intermediate: {intermediate_size}\n")

    all_results = {}
    completion_stats = {}
    overall_start = time.time()
    prompt_idx = 0

    for category, prompts in all_prompts.items():
        print(f"\n=== Category: {category} ({len(prompts)} prompts) ===")
        cat_tracker = defaultdict(lambda: defaultdict(lambda: {
            "wnorm": 0.0, "rnorm": 0.0, "wsum": 0.0, "tc": 0, "cc": 0,
            "neuron_act": torch.zeros(intermediate_size),
        }))
        n_completed = 0
        n_filtered = 0

        for i, prompt in enumerate(prompts):
            prompt_idx += 1
            full_seq, plen, glen, hit_max, gen_t = phase1_generate(
                model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)

            if hit_max:
                # Filter: don't include incomplete prompts
                n_filtered += 1
                elapsed = (time.time() - overall_start) / 60
                eta = elapsed / prompt_idx * (total_prompts - prompt_idx)
                print(f"  [{prompt_idx}/{total_prompts}] {category} #{i+1}: "
                      f"FILTERED (hit max {glen} tok in {gen_t:.0f}s) "
                      f"elapsed {elapsed:.0f}m ETA {eta:.0f}m")
                continue

            tracker, prof_t = phase2_profile(
                model, full_seq, num_layers, num_experts, intermediate_size)
            merge_trackers(cat_tracker, tracker, num_layers, num_experts)
            n_completed += 1

            elapsed = (time.time() - overall_start) / 60
            eta = elapsed / prompt_idx * (total_prompts - prompt_idx)
            print(f"  [{prompt_idx}/{total_prompts}] {category} #{i+1}: "
                  f"OK {glen}t {gen_t:.0f}s + prof {prof_t:.0f}s "
                  f"(seq {full_seq.shape[0]}, elapsed {elapsed:.0f}m ETA {eta:.0f}m)")

        completion_stats[category] = {
            "completed": n_completed,
            "filtered": n_filtered,
            "total": len(prompts),
        }
        all_results[category] = cat_tracker

    print(f"\n=== Completion Stats ===")
    for cat, stats in completion_stats.items():
        print(f"  {cat}: {stats['completed']}/{stats['total']} completed, "
              f"{stats['filtered']} filtered")

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    save_data = {
        "metadata": {
            "model": MODEL_PATH,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "intermediate_size": intermediate_size,
            "hidden_size": hidden_size,
            "max_new_tokens": MAX_NEW_TOKENS,
            "completion_stats": completion_stats,
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
    print(f"File size: {os.path.getsize(OUTPUT_FILE)/1024**2:.0f} MB")


if __name__ == "__main__":
    main()
