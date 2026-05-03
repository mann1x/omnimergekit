#!/usr/bin/env python3
"""
Expert contribution analysis v2 — two-phase approach for CPU speed.

Phase 1: Generate responses with lightweight router hooks (routing weights only).
Phase 2: Single forward pass of full prompt+response with heavy hooks (expert output norms).

This avoids recomputing expert outputs during autoregressive generation (128 steps)
and only does it once in the forward pass (1 step).
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

TOPICS = {
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

MODEL_PATH = "."


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading fp16 on CPU...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    print(f"Loaded in {time.time()-t0:.0f}s")
    return model, tokenizer


def phase1_generate(model, tokenizer, prompt, max_new_tokens=128):
    """Generate response. No heavy hooks — just generate and return full sequence."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True,
        add_generation_prompt=True, enable_thinking=True)
    input_ids = inputs["input_ids"]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    gen_len = out.shape[1] - input_ids.shape[1]
    elapsed = time.time() - t0
    return out[0], input_ids.shape[1], gen_len, elapsed


def phase2_profile(model, full_sequence, num_layers, num_experts):
    """Single forward pass of full sequence with expert norm hooks.
    This runs once (not autoregressive), so hooks only fire once per layer."""
    tracker = defaultdict(lambda: defaultdict(
        lambda: {"wnorm": 0.0, "rnorm": 0.0, "wsum": 0.0, "tc": 0, "cc": 0}))
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
                        g, u = nn.functional.linear(cs, module.gate_up_proj[eidx]).chunk(2, dim=-1)
                        eh = module.act_fn(g) * u
                        eh = nn.functional.linear(eh, module.down_proj[eidx])
                        wt = top_k_wt[tidx, pos]
                        weighted = eh * wt.unsqueeze(-1)
                        tracker[layer_idx][eid]["wnorm"] += weighted.norm().item()
                        tracker[layer_idx][eid]["rnorm"] += eh.norm().item()
                        tracker[layer_idx][eid]["wsum"] += wt.mean().item()
                        tracker[layer_idx][eid]["tc"] += len(tidx)
                        tracker[layer_idx][eid]["cc"] += 1
            return hook

        hooks.append(layer.experts.register_forward_hook(make_hook(li)))

    # Single forward pass — all tokens at once, not autoregressive
    input_ids = full_sequence.unsqueeze(0)
    mm_ids = torch.zeros_like(input_ids)
    t0 = time.time()
    with torch.no_grad():
        model(input_ids, mm_token_type_ids=mm_ids)
    elapsed = time.time() - t0

    for h in hooks:
        h.remove()

    return tracker, elapsed


def analyze(tracker, num_layers, num_experts):
    """Convert tracker to structured results."""
    result = {}
    for li in range(num_layers):
        tw = sum(tracker[li][eid]["wnorm"] for eid in range(num_experts))
        experts = []
        for eid in range(num_experts):
            d = tracker[li][eid]
            experts.append({
                "id": eid, "wnorm": d["wnorm"], "rnorm": d["rnorm"],
                "avgw": d["wsum"] / d["cc"] if d["cc"] > 0 else 0,
                "tc": d["tc"],
                "pct": d["wnorm"] / max(tw, 1e-10) * 100,
            })
        experts.sort(key=lambda x: x["wnorm"], reverse=True)
        result[li] = {"tw": tw, "experts": experts}
    return result


def print_results(all_results, num_layers, num_experts):
    print(f"\n{'='*90}")
    print(f"  EXPERT CONTRIBUTION ANALYSIS (weighted output norm)")
    print(f"{'='*90}")

    # Concentration per topic
    print(f"\n  Contribution concentration per topic:")
    print(f"    {'Topic':>10}  {'Top-1%':>8}  {'Top-8%':>8}  {'Top-32%':>8}  "
          f"{'Gini':>6}  {'80%need':>7}")
    for topic in TOPICS:
        et = defaultdict(float)
        for li in range(num_layers):
            for e in all_results[topic][li]["experts"]:
                et[e["id"]] += e["wnorm"]
        total = sum(et.values())
        if total == 0:
            continue
        sc = sorted(et.values(), reverse=True)
        cs = np.cumsum(sc) / total * 100
        n80 = int(np.searchsorted(cs, 80)) + 1
        sa = sorted(sc)
        n = len(sa)
        gini = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sa)) / (n * total)
        print(f"    {topic:>10}  {sc[0]/total*100:>7.1f}%  {sum(sc[:8])/total*100:>7.1f}%  "
              f"{sum(sc[:32])/total*100:>7.1f}%  {gini:>6.3f}  {n80:>5}")

    # Per-layer (math)
    print(f"\n  Per-layer (math):")
    print(f"    {'Layer':>5}  {'Top-1':>12}  {'Top1%':>6}  {'Top8%':>6}  {'Act':>4}  {'Bot64%':>7}")
    for li in range(num_layers):
        d = all_results["math"][li]
        tw = d["tw"]
        if tw == 0:
            continue
        e = d["experts"]
        t1 = e[0]
        t8 = sum(x["wnorm"] for x in e[:8])
        b64 = sum(x["wnorm"] for x in e[64:])
        act = sum(1 for x in e if x["tc"] > 0)
        print(f"    {li:>5}  e{t1['id']:>3d}({t1['avgw']:.3f})  {t1['pct']:>5.1f}%  "
              f"{t8/tw*100:>5.1f}%  {act:>4}  {b64/tw*100:>6.1f}%")

    # Math vs creative
    mt = defaultdict(float)
    ct = defaultdict(float)
    for li in range(num_layers):
        for e in all_results["math"][li]["experts"]:
            mt[e["id"]] += e["wnorm"]
        for e in all_results["creative"][li]["experts"]:
            ct[e["id"]] += e["wnorm"]
    ms = sorted(mt.items(), key=lambda x: x[1], reverse=True)
    cs_ = sorted(ct.items(), key=lambda x: x[1], reverse=True)
    mtot = sum(v for _, v in ms)
    ctot = sum(v for _, v in cs_)

    print(f"\n  Top-10: math vs creative")
    for r in range(10):
        me, mv = ms[r]
        ce, cv = cs_[r]
        print(f"    #{r+1:>2}  e{me:>3d} {mv/mtot*100:>5.1f}%   e{ce:>3d} {cv/ctot*100:>5.1f}%")

    m32 = set(e for e, _ in ms[:32])
    c32 = set(e for e, _ in cs_[:32])
    print(f"\n  Top-32 overlap: {len(m32 & c32)}/32")

    # Per-layer top-8 overlap
    print(f"\n  Per-layer top-8 overlap (math vs creative):")
    for li in range(num_layers):
        me = {e["id"]: e["wnorm"] for e in all_results["math"][li]["experts"]}
        ce = {e["id"]: e["wnorm"] for e in all_results["creative"][li]["experts"]}
        m8 = set(sorted(me, key=me.get, reverse=True)[:8])
        c8 = set(sorted(ce, key=ce.get, reverse=True)[:8])
        ovl = len(m8 & c8)
        print(f"    L{li:2d}: {ovl}/8, math-only={sorted(m8-c8)}, creative-only={sorted(c8-m8)}")


def main():
    model, tokenizer = load_model()
    num_layers = model.config.text_config.num_hidden_layers
    num_experts = model.config.text_config.num_experts
    print(f"{num_layers} layers, {num_experts} experts\n")

    all_results = {}

    for topic, prompts in TOPICS.items():
        print(f"Topic: {topic}")
        topic_tracker = defaultdict(lambda: defaultdict(
            lambda: {"wnorm": 0.0, "rnorm": 0.0, "wsum": 0.0, "tc": 0, "cc": 0}))

        for i, prompt in enumerate(prompts):
            # Phase 1: Generate (fast, no hooks)
            full_seq, prompt_len, gen_len, gen_time = phase1_generate(
                model, tokenizer, prompt, max_new_tokens=128)
            print(f"  [{i+1}/{len(prompts)}] gen: {gen_len} tok in {gen_time:.0f}s", end="")

            # Phase 2: Profile full sequence (one forward pass with hooks)
            tracker, prof_time = phase2_profile(model, full_seq, num_layers, num_experts)
            print(f", profile: {prof_time:.0f}s, total: {full_seq.shape[0]} tok")

            # Accumulate into topic tracker
            for li in range(num_layers):
                for eid in range(num_experts):
                    for key in ["wnorm", "rnorm", "wsum", "tc", "cc"]:
                        topic_tracker[li][eid][key] += tracker[li][eid][key]

        all_results[topic] = analyze(topic_tracker, num_layers, num_experts)

    print_results(all_results, num_layers, num_experts)

    # Save ALL experts per layer (not just top 20)
    os.makedirs("eval_results", exist_ok=True)
    save = {}
    for topic, result in all_results.items():
        save[topic] = {
            str(li): {"tw": d["tw"], "experts": d["experts"]}
            for li, d in result.items()
        }
    with open("eval_results/expert_contributions_full.json", "w") as f:
        json.dump(save, f, indent=2)
    print("\nSaved to eval_results/expert_contributions_full.json")

    # Per-layer drop candidate map at various thresholds
    print(f"\n{'='*90}")
    print(f"  PER-LAYER DROP CANDIDATES")
    print(f"{'='*90}")

    # Aggregate across all topics
    agg = defaultdict(lambda: defaultdict(float))
    for topic in TOPICS:
        for li in range(num_layers):
            for e in all_results[topic][li]["experts"]:
                agg[li][e["id"]] += e["wnorm"]

    thresholds = [0.90, 0.95, 0.99]
    print(f"\n  Experts needed to reach X% of total contribution:")
    print(f"    {'Layer':>5}  {'Active':>6}  {'90%':>5}  {'95%':>5}  {'99%':>5}  "
          f"{'Drop@95%':>8}  {'Drop@99%':>8}")
    total_drop_95 = 0
    total_drop_99 = 0
    for li in range(num_layers):
        experts = agg[li]
        total = sum(experts.values())
        if total == 0:
            continue
        sorted_e = sorted(experts.items(), key=lambda x: x[1], reverse=True)
        active = sum(1 for _, v in sorted_e if v > 0)
        cumsum = np.cumsum([v for _, v in sorted_e]) / total
        needs = {}
        for t in thresholds:
            needs[t] = int(np.searchsorted(cumsum, t)) + 1
        drop_95 = num_experts - needs[0.95]
        drop_99 = num_experts - needs[0.99]
        total_drop_95 += drop_95
        total_drop_99 += drop_99
        print(f"    {li:>5}  {active:>6}  {needs[0.90]:>5}  {needs[0.95]:>5}  {needs[0.99]:>5}  "
              f"{drop_95:>8}  {drop_99:>8}")

    print(f"\n    Total droppable experts across all layers:")
    print(f"      At 95%: {total_drop_95}/{num_experts*num_layers} "
          f"({total_drop_95/num_experts/num_layers*100:.1f}%)")
    print(f"      At 99%: {total_drop_99}/{num_experts*num_layers} "
          f"({total_drop_99/num_experts/num_layers*100:.1f}%)")

    # Per-layer drop lists at 99% threshold
    drop_map = {}
    print(f"\n  Drop candidates per layer (99% threshold):")
    for li in range(num_layers):
        experts = agg[li]
        total = sum(experts.values())
        if total == 0:
            continue
        sorted_e = sorted(experts.items(), key=lambda x: x[1], reverse=True)
        cumsum = np.cumsum([v for _, v in sorted_e]) / total
        keep_n = int(np.searchsorted(cumsum, 0.99)) + 1
        keep_ids = set(eid for eid, _ in sorted_e[:keep_n])
        drop_ids = sorted(set(range(num_experts)) - keep_ids)
        drop_map[li] = drop_ids
        print(f"    L{li:2d}: keep {keep_n:3d}, drop {len(drop_ids):3d} — "
              f"drop examples: {drop_ids[:10]}{'...' if len(drop_ids)>10 else ''}")

    # What's the minimum experts per layer we can target?
    min_keep = min(num_experts - len(v) for v in drop_map.values())
    max_keep = max(num_experts - len(v) for v in drop_map.values())
    print(f"\n  Keep range: {min_keep}-{max_keep} experts per layer at 99% threshold")

    # Save drop map
    with open("eval_results/expert_drop_map.json", "w") as f:
        json.dump({str(li): ids for li, ids in drop_map.items()}, f, indent=2)
    print(f"  Saved drop map to eval_results/expert_drop_map.json")


if __name__ == "__main__":
    main()
