#!/usr/bin/env python3
"""expert_neuron_analysis_v5_targeted.py — T17.3 mapping run for v5-code (and v5-science / v5-math).

NOTE: separate from the older `expert_neuron_analysis_v5.py` (Apr 8) which was a
prompt-set exploration unrelated to the T17 targeted-variants strategy. This is
the canonical implementation for the T17 design (`docs/T17_v5_targeted_pruning_strategy.md`).

Two-tier signal:

  • Tier-A — generic synthetic prompts (math/logic/code/science/creative,
    8 each, 128 new tokens, greedy on 128e). Identical to v4 — broad-competence
    floor. Re-generated fresh each run.

  • Tier-B — PASS-trace replay. For each 128e PASS trace from the variant's
    target benchmarks (v5-code = HE + HE+ + LCB-medium-55), we tokenize the
    saved `prompt + completion` and replay it through 128e with forward hooks.
    No generation needed — the completions already exist on disk. Traces
    longer than `--window-tokens` are split into overlapping chunks.

Output: `scripts/expert_neuron_v5_<variant>.json` — drop-in compatible with the
v4 JSON schema for `generate_drop_map_multiclass.py` extension, with 3 extra
`targeted_*` class keys alongside the 5 generic ones.

Runtime estimate (RTX 3090, BF16, eager attention):
  • Tier-A: 40 prompts × 128 tok generate + replay  ≈  30-40 min
  • Tier-B: 360 traces, avg ~1100 tok each = ~400k tokens of replay
            at ~5-10 tok/sec on Gemma 4 26B-A4B  ≈  11-22 hr
  • Total: ~12-22 hr (overnight)

REQUIREMENT: must run inside `/shared/dev/lightseek/.venv/bin/python` for
transformers 5.5.0 (Gemma 4 support).
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_TOKEN",
    open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    if Path("~/.cache/huggingface/token").expanduser().exists() else "")

import torch                                                          # noqa: E402
from torch import nn                                                  # noqa: E402

WS = Path("/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models")
MODEL_PATH = WS / "google" / "gemma-4-26B-A4B-it"

# Tier-A: 40 synthetic prompts (verbatim from expert_neuron_analysis_v4.py — same
# distribution that produced the working 109e/98e v3/v4 priors). DO NOT alter
# these without re-running v4 for comparability.
TIER_A_PROMPTS = {
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


def _new_expert_entry(intermediate_size: int) -> dict:
    return {
        "wnorm": 0.0,
        "rnorm": 0.0,
        "wsum": 0.0,
        "tc": 0,
        "cc": 0,
        "neuron_act": torch.zeros(intermediate_size, dtype=torch.float64),
    }


def _new_per_layer_tracker(num_layers: int, num_experts: int, intermediate_size: int):
    return [
        [_new_expert_entry(intermediate_size) for _ in range(num_experts)]
        for _ in range(num_layers)
    ]


def make_hooks(model, num_layers: int, tracker: list[list[dict]], weight: float):
    """Install hooks on each MoE `experts` module. Accumulates into `tracker`
    weighted by `weight` (Set-B traces use weight=3.0 per T17 design).

    Gemma 4 flattens to [B*T, D] before experts() — top_k_idx is [B*T, top_k].
    """
    hooks = []
    for li in range(num_layers):
        layer = model.model.language_model.layers[li]
        if not hasattr(layer, "experts"):
            continue

        def make_hook(layer_idx: int):
            def hook(module, args, output):
                hs, top_k_idx, top_k_wt = args
                n = module.num_experts
                with torch.no_grad():
                    mask = nn.functional.one_hot(top_k_idx, num_classes=n).permute(2, 1, 0)
                    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
                    for eidx_t in hit:
                        eidx = int(eidx_t[0])
                        if eidx == n:
                            continue
                        pos, tidx = torch.where(mask[eidx])
                        if tidx.numel() == 0:
                            continue
                        cs = hs[tidx]
                        gu = nn.functional.linear(cs, module.gate_up_proj[eidx])
                        g, u = gu.chunk(2, dim=-1)
                        intermediate = module.act_fn(g) * u
                        neuron_sq = (intermediate.float() ** 2).sum(dim=0)
                        tracker[layer_idx][eidx]["neuron_act"] += (
                            neuron_sq.cpu().to(torch.float64) * weight)
                        eh = nn.functional.linear(intermediate, module.down_proj[eidx])
                        wt = top_k_wt[tidx, pos]
                        weighted = eh * wt.unsqueeze(-1)
                        tracker[layer_idx][eidx]["wnorm"] += float(weighted.norm().item()) * weight
                        tracker[layer_idx][eidx]["rnorm"] += float(eh.norm().item()) * weight
                        tracker[layer_idx][eidx]["wsum"]  += float(wt.float().mean().item()) * weight
                        tracker[layer_idx][eidx]["tc"]    += int(tidx.numel() * weight)
                        tracker[layer_idx][eidx]["cc"]    += int(1 * weight)
            return hook
        hooks.append(layer.experts.register_forward_hook(make_hook(li)))
    return hooks


def forward_pass(model, input_ids: torch.Tensor):
    mm_ids = torch.zeros_like(input_ids)
    with torch.no_grad():
        model(input_ids, mm_token_type_ids=mm_ids)


def tier_a_run(model, tokenizer, num_layers, num_experts, intermediate_size,
               max_new_tokens, all_cats: dict, done_keys: set,
               checkpoint_cb=None) -> None:
    """Accumulates into `all_cats[f"generic_{cat}"]` in-place. Skips items
    in `done_keys` (strings `"tier_a/<cat>/<prompt_idx>"`). Calls
    `checkpoint_cb(done_keys)` after each prompt so progress is durable."""
    total = sum(len(v) for v in TIER_A_PROMPTS.values())
    overall_t0 = time.time()
    idx = 0
    for cat, prompts in TIER_A_PROMPTS.items():
        cat_key = f"generic_{cat}"
        # Resume: use existing tracker if present (mid-class crash), else fresh
        if cat_key in all_cats:
            cat_tracker = all_cats[cat_key]
            print(f"\n[Tier-A] === {cat} === (RESUMING)", flush=True)
        else:
            cat_tracker = _new_per_layer_tracker(num_layers, num_experts, intermediate_size)
            all_cats[cat_key] = cat_tracker
            print(f"\n[Tier-A] === {cat} ===", flush=True)
        for pi, prompt in enumerate(prompts):
            idx += 1
            item_key = f"tier_a/{cat}/{pi}"
            if item_key in done_keys:
                print(f"  [{idx}/{total}] {cat} #{pi+1}: SKIP (in checkpoint)", flush=True)
                continue
            msgs = [{"role": "user", "content": prompt}]
            t0 = time.time()
            inputs = tokenizer.apply_chat_template(
                msgs, return_tensors="pt", return_dict=True,
                add_generation_prompt=True, enable_thinking=True)
            input_ids = inputs["input_ids"].to(model.device)
            with torch.no_grad():
                gen = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            # Council bug #1 fix (csl-2026-05-15-1439-4ff0): some HF generate
            # paths (Gemma 4 + bf16/CUDA + SDPA) returned only the newly-
            # generated tokens, dropping the prompt. That halved tc and
            # produced a 70%-divergent importance map vs the v4 fp16/CPU
            # reference. Detect both layouts and always profile on the FULL
            # sequence (prompt + completion).
            gen_row = gen[0]
            prompt_len = input_ids.shape[1]
            if gen_row.shape[0] >= prompt_len and bool(
                    torch.equal(gen_row[:prompt_len].to(input_ids.device),
                                input_ids[0])):
                full_seq = gen_row  # already prompt+completion
                layout = "prompt+gen"
            else:
                full_seq = torch.cat(
                    [input_ids[0], gen_row.to(input_ids.device)], dim=0)
                layout = "gen-only-fixed"
            if idx <= 2:
                print(f"  [debug v5-fix#1] prompt_len={prompt_len} "
                      f"gen_row_len={gen_row.shape[0]} "
                      f"full_seq_len={full_seq.shape[0]} layout={layout}",
                      flush=True)
            gen_t = time.time() - t0
            t1 = time.time()
            hooks = make_hooks(model, num_layers, cat_tracker, weight=1.0)
            try:
                forward_pass(model, full_seq.unsqueeze(0))
            finally:
                for h in hooks:
                    h.remove()
            prof_t = time.time() - t1
            elapsed = (time.time() - overall_t0) / 60
            eta = elapsed / idx * (total - idx) if idx else 0
            print(f"  [{idx}/{total}] {cat} #{pi+1}: gen {gen_t:.0f}s + prof {prof_t:.0f}s "
                  f"(elapsed {elapsed:.0f}m ETA Tier-A {eta:.0f}m)", flush=True)
            done_keys.add(item_key)
            if checkpoint_cb is not None:
                checkpoint_cb(done_keys)


def chunk_input(input_ids: torch.Tensor, window: int, overlap: int):
    T = input_ids.shape[1]
    if T <= window:
        yield input_ids
        return
    step = max(window - overlap, 1)
    start = 0
    while start < T:
        end = min(start + window, T)
        yield input_ids[:, start:end]
        if end == T:
            break
        start += step


def tier_b_run(model, tokenizer, num_layers, num_experts, intermediate_size,
               traces, window, overlap, all_cats: dict, done_keys: set,
               checkpoint_cb=None) -> None:
    """Accumulates per-bench trackers into `all_cats[f"targeted_{bench}"]`
    in-place, skipping traces in `done_keys` (strings
    `"tier_b/<bench>/<task_id>/<set>"`). Calls `checkpoint_cb(done_keys)`
    after each trace."""
    overall_t0 = time.time()
    total = len(traces)
    for ti, trace in enumerate(traces):
        bench = trace["bench"]
        weight = float(trace.get("weight", 1.0))
        item_key = f"tier_b/{bench}/{trace['task_id']}/{trace.get('set','?')}"
        if item_key in done_keys:
            if ti % 10 == 0:
                print(f"  [{ti+1}/{total}] {bench}/{trace['task_id']} SKIP (ckpt)",
                      flush=True)
            continue
        prompt = trace["prompt"]
        completion = trace["completion"]
        msgs = [{"role": "user", "content": prompt}]
        chat_ids = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", return_dict=True,
            add_generation_prompt=True, enable_thinking=True)["input_ids"]
        comp_ids = tokenizer(completion, return_tensors="pt",
                             add_special_tokens=False)["input_ids"]
        full_ids = torch.cat([chat_ids, comp_ids], dim=-1).to(model.device)
        bench_key = f"targeted_{bench}"
        if bench_key not in all_cats:
            all_cats[bench_key] = _new_per_layer_tracker(num_layers, num_experts, intermediate_size)
        tracker = all_cats[bench_key]
        n_chunks = 0
        t0 = time.time()
        hooks = make_hooks(model, num_layers, tracker, weight=weight)
        try:
            for chunk_ids in chunk_input(full_ids, window, overlap):
                n_chunks += 1
                forward_pass(model, chunk_ids)
        finally:
            for h in hooks:
                h.remove()
        dt = time.time() - t0
        elapsed = (time.time() - overall_t0) / 60
        eta = elapsed / (ti + 1) * (total - ti - 1) if (ti + 1) else 0
        if ti % 10 == 0 or ti == total - 1:
            print(f"  [{ti+1}/{total}] {bench}/{trace['task_id']} set={trace['set']} "
                  f"wt={weight:.1f} len={full_ids.shape[1]} chunks={n_chunks} "
                  f"dt={dt:.1f}s (elapsed {elapsed:.0f}m ETA {eta:.0f}m)", flush=True)
        done_keys.add(item_key)
        if checkpoint_cb is not None:
            checkpoint_cb(done_keys)


def serialize(categories: dict[str, list], num_layers: int, num_experts: int) -> dict:
    out = {}
    for cat_name, tracker in categories.items():
        cat = {}
        for li in range(num_layers):
            rows = []
            for eid in range(num_experts):
                d = tracker[li][eid]
                rows.append({
                    "id": eid,
                    "wnorm": float(d["wnorm"]),
                    "rnorm": float(d["rnorm"]),
                    "wsum":  float(d["wsum"]),
                    "tc":    int(d["tc"]),
                    "cc":    int(d["cc"]),
                    "neuron_act": d["neuron_act"].tolist(),
                })
            cat[str(li)] = rows
        out[cat_name] = cat
    return out


# ── Checkpointing (added 2026-05-14 after the 7-h-Tier-A loss scare) ───────
# Both Tier-A and Tier-B accumulate large in-memory trackers and write the
# final JSON only at the end of main(). A crash anywhere before serialize()
# loses all prior compute. Checkpoint after EACH prompt/trace so the cost
# of any single crash is bounded to one item. Resume on restart by loading
# `<out>.checkpoint.json` and skipping items in the `_state.done` set.

def _deserialize_tracker(cat_rows: dict, num_layers: int, num_experts: int,
                          intermediate_size: int) -> list:
    """Inverse of `serialize()` for a single category — list-of-list-of-dict
    with torch tensors, ready to keep accumulating."""
    tracker = _new_per_layer_tracker(num_layers, num_experts, intermediate_size)
    for li_str, rows in cat_rows.items():
        li = int(li_str)
        for r in rows:
            eid = r["id"]
            entry = tracker[li][eid]
            entry["wnorm"] = float(r["wnorm"])
            entry["rnorm"] = float(r["rnorm"])
            entry["wsum"]  = float(r["wsum"])
            entry["tc"]    = int(r["tc"])
            entry["cc"]    = int(r["cc"])
            entry["neuron_act"] = torch.tensor(r["neuron_act"], dtype=torch.float64)
    return tracker


def _checkpoint_write(ckpt_path: Path, all_cats: dict, state: dict,
                       num_layers: int, num_experts: int):
    """Atomic write: serialize trackers + state to a tmp file then rename."""
    payload = {
        "_state": state,
        "categories": serialize(all_cats, num_layers, num_experts),
    }
    tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, ckpt_path)


def _checkpoint_load(ckpt_path: Path, num_layers: int, num_experts: int,
                      intermediate_size: int):
    """Load checkpoint, returning (all_cats, state) or (None, None) if absent
    or malformed. all_cats holds in-memory torch trackers ready to continue."""
    if not ckpt_path.exists():
        return None, None
    try:
        with open(ckpt_path) as f:
            payload = json.load(f)
        all_cats = {}
        for cat_name, cat_rows in payload.get("categories", {}).items():
            all_cats[cat_name] = _deserialize_tracker(
                cat_rows, num_layers, num_experts, intermediate_size)
        state = payload.get("_state", {})
        return all_cats, state
    except Exception as e:
        print(f"[checkpoint] WARN: failed to load {ckpt_path}: {e}", flush=True)
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["code", "science", "math", "logic"], required=True)
    ap.add_argument("--tier-b-json", required=True,
                    help="Output of extract_pass_traces.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier-a-max-tokens", type=int, default=128)
    ap.add_argument("--window-tokens", type=int, default=2048)
    # Council bug #2 fix (csl-2026-05-15-1439-4ff0): overlap>0 causes tokens
    # in the overlap regions to be hooked TWICE on consecutive chunks,
    # systematically inflating per-expert scores in the overlap zones and
    # corrupting Tier-B targeted maps. Default lowered from 256 → 0.
    ap.add_argument("--window-overlap", type=int, default=0,
                    help="Token overlap between consecutive chunks for Tier-B "
                         "replays. MUST be 0 unless paired with a deduplication "
                         "mask in the hooks (not implemented). Default 0.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--skip-tier-a", action="store_true",
                    help="Skip Tier-A computation (only valid if Tier-A data is already "
                         "in the checkpoint or loaded via --load-tier-a-from).")
    ap.add_argument("--load-tier-a-from", default=None,
                    help="Import generic_* categories from an existing v5 output JSON. "
                         "Use when running additional variants (science/math) after the "
                         "first (code) so we don't recompute the ~7.5h Tier-A. Implies "
                         "--skip-tier-a unless that flag is explicitly set. The loaded "
                         "Tier-A data is verified against the current model's "
                         "num_layers/num_experts/intermediate_size before use.")
    ap.add_argument("--quant", default="bf16", choices=["bf16", "nf4"],
                    help="bf16: full BF16 weights (needs ≥48 GB GPU OR CPU offload thrash); "
                         "nf4: bitsandbytes 4-bit NF4 (fits 24 GB GPU, ~5-10× faster, "
                         "small signal noise on wnorm/tc).")
    args = ap.parse_args()

    print(f"=== expert_neuron_analysis_v5_targeted — variant={args.variant} ===", flush=True)
    print(f"  model: {MODEL_PATH}", flush=True)
    print(f"  device: {args.device}  dtype: {args.dtype}", flush=True)
    print(f"  tier-a max_new_tokens: {args.tier_a_max_tokens}", flush=True)
    print(f"  tier-b window/overlap: {args.window_tokens}/{args.window_overlap}", flush=True)

    print(f"\nLoading Tier-B traces from {args.tier_b_json} …", flush=True)
    with open(args.tier_b_json) as f:
        tier_b = json.load(f)
    traces = tier_b["traces"]
    print(f"  {len(traces)} traces, set counts: {tier_b['metadata']['set_counts']}", flush=True)

    print("\nLoading 128e model …", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if args.device == "cuda" and args.quant == "nf4":
        # bitsandbytes NF4 4-bit: ~13 GB on GPU for Gemma 4 26B-A4B, leaving
        # ~10 GB for activations + hook tensors on a 3090. Compute dtype is
        # bf16 — wnorm/tc signals are measured from dequantized weights
        # (small noise vs BF16, but rank ordering preserved on the scales
        # that matter for the drop map).
        # transformers 5.5/5.8 + bnb 0.49.2 had a 2-stage incompatibility
        # (Params4bit.__new__ rejecting accelerate's _is_hf_initialized, and
        # _save_to_state_dict calling .item() on meta-tensor QuantState.offset
        # during accelerate's len()-probe of state_dict). Both are fixed
        # durably in the forks at /shared/dev/{transformers,bitsandbytes}
        # (branches v5.5.0 and 0.49.2, installed editable). The inline
        # monkey-patch has been removed; verify the forks are loaded via
        #   python -c "import transformers, bitsandbytes; print(transformers.__file__, bitsandbytes.__file__)"
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,  # allow non-bnb layers
                                                    # (embed, lm_head) on CPU
        )
        # device_map="auto" lets transformers stream each shard through bnb
        # compression — putting `"": 0` materializes the whole 47 GB first
        # shard on GPU before bnb gets a chance to compress → OOM.
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH), quantization_config=bnb_cfg,
            device_map="auto",
            max_memory={0: "22GiB", "cpu": "200GiB"},
            trust_remote_code=True,
            low_cpu_mem_usage=True, attn_implementation="sdpa")
    elif args.device == "cuda":
        # 26B-A4B BF16 = ~52 GB; 3090 = 24 GB. device_map="auto" spills cold
        # layers to CPU and swaps them JIT — SLOW (per-token thrash). Use
        # --quant nf4 for ~5-10× speedup at small signal cost.
        _gpu_budget_gib = 18
        max_memory = {0: f"{_gpu_budget_gib}GiB", "cpu": "200GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH), dtype=dtype, device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True, low_cpu_mem_usage=True,
            attn_implementation="sdpa")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH), dtype=dtype, device_map="cpu",
            trust_remote_code=True, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model.eval()
    num_layers = model.config.text_config.num_hidden_layers
    num_experts = model.config.text_config.num_experts
    intermediate_size = model.config.text_config.moe_intermediate_size
    hidden_size = model.config.text_config.hidden_size
    print(f"  loaded in {time.time()-t0:.0f}s — layers={num_layers} "
          f"experts={num_experts} intermediate={intermediate_size}", flush=True)

    # ── Checkpoint setup ──────────────────────────────────────────────────
    out_path = Path(args.out)
    ckpt_path = out_path.with_suffix(out_path.suffix + ".checkpoint.json")
    all_cats, ckpt_state = _checkpoint_load(
        ckpt_path, num_layers, num_experts, intermediate_size)
    if all_cats is None:
        all_cats = {}
        done_keys: set = set()
        print(f"[checkpoint] starting fresh (no {ckpt_path.name} found)", flush=True)
    else:
        done_keys = set(ckpt_state.get("done", []))
        print(f"[checkpoint] LOADED — {len(all_cats)} cats, {len(done_keys)} items done",
              flush=True)

    def _make_state(phase: str) -> dict:
        return {
            "phase": phase,
            "variant": args.variant,
            "done": sorted(done_keys),
            "tier_b_source": args.tier_b_json,
            "tier_b_trace_count": len(traces),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _ckpt_cb_a(dk):
        _checkpoint_write(ckpt_path, all_cats, _make_state("tier_a"),
                          num_layers, num_experts)

    def _ckpt_cb_b(dk):
        _checkpoint_write(ckpt_path, all_cats, _make_state("tier_b"),
                          num_layers, num_experts)

    # Best-effort dump on SIGTERM/SIGINT so a kill still saves partial state
    import signal as _sig
    def _on_signal(signum, frame):
        try:
            _checkpoint_write(ckpt_path, all_cats, _make_state(f"signal_{signum}"),
                              num_layers, num_experts)
            print(f"[checkpoint] flushed on signal {signum} → {ckpt_path}", flush=True)
        finally:
            raise SystemExit(128 + signum)
    _sig.signal(_sig.SIGTERM, _on_signal)
    _sig.signal(_sig.SIGINT,  _on_signal)

    # ── Optional: import Tier-A from an existing v5 output ────────────────
    # Lets v5-science / v5-math reuse the v5-code Tier-A (~7.5h on solidPC
    # 3090). Validates layer/expert/intermediate-size match before merging.
    if args.load_tier_a_from:
        src_path = Path(args.load_tier_a_from)
        if not src_path.exists():
            raise SystemExit(f"--load-tier-a-from: {src_path} does not exist")
        print(f"\n[load-tier-a-from] importing generic_* from {src_path}", flush=True)
        with open(src_path) as f:
            src = json.load(f)
        src_meta = src.get("metadata", {})
        if (int(src_meta.get("num_layers", -1)) != num_layers
                or int(src_meta.get("num_experts", -1)) != num_experts
                or int(src_meta.get("intermediate_size", -1)) != intermediate_size):
            raise SystemExit(
                f"--load-tier-a-from: dimension mismatch — source has "
                f"L={src_meta.get('num_layers')} E={src_meta.get('num_experts')} "
                f"I={src_meta.get('intermediate_size')}, current model has "
                f"L={num_layers} E={num_experts} I={intermediate_size}")
        n_imported = 0
        for cat_name, cat_rows in src.get("categories", {}).items():
            if not cat_name.startswith("generic_"):
                continue  # only Tier-A categories
            if cat_name in all_cats:
                print(f"  {cat_name}: already in checkpoint, skipping import", flush=True)
                continue
            all_cats[cat_name] = _deserialize_tracker(
                cat_rows, num_layers, num_experts, intermediate_size)
            n_imported += 1
        # Seed done_keys for every Tier-A prompt so any subsequent tier_a_run
        # (e.g. if user didn't pass --skip-tier-a) is a strict no-op.
        for cat, prompts in TIER_A_PROMPTS.items():
            for pi in range(len(prompts)):
                done_keys.add(f"tier_a/{cat}/{pi}")
        print(f"[load-tier-a-from] imported {n_imported} categories, "
              f"seeded {sum(len(p) for p in TIER_A_PROMPTS.values())} Tier-A done keys",
              flush=True)
        # Persist immediately so a crash before Tier-B writes still has Tier-A
        _checkpoint_write(ckpt_path, all_cats, _make_state("tier_a_imported"),
                          num_layers, num_experts)
        # If user did not also pass --skip-tier-a, default to skipping anyway
        # (otherwise tier_a_run would just no-op against the seeded done_keys
        # but waste a model.generate() call per prompt before checking).
        if not args.skip_tier_a:
            print("[load-tier-a-from] implies --skip-tier-a (auto-enabled)", flush=True)
            args.skip_tier_a = True

    if not args.skip_tier_a:
        print("\n=== Tier-A: 40 synthetic prompts ===", flush=True)
        tier_a_run(model, tokenizer, num_layers, num_experts,
                   intermediate_size, args.tier_a_max_tokens,
                   all_cats, done_keys, checkpoint_cb=_ckpt_cb_a)
    else:
        # Sanity: Tier-A must be present in all_cats for the final output to
        # contain generic_* categories. Hard-fail early if missing — better
        # than running Tier-B for ~100m and discovering incomplete output.
        missing = [f"generic_{cat}" for cat in TIER_A_PROMPTS
                   if f"generic_{cat}" not in all_cats]
        if missing:
            raise SystemExit(
                f"[skip-tier-a] no Tier-A data in checkpoint or loaded source; "
                f"missing categories: {missing}. Use --load-tier-a-from "
                f"<existing-v5-output.json> or remove --skip-tier-a.")
        print("\n[skip-tier-a] Tier-A present in all_cats — proceeding to Tier-B.",
              flush=True)

    print(f"\n=== Tier-B: {len(traces)} PASS-trace replays ===", flush=True)
    tier_b_run(model, tokenizer, num_layers, num_experts,
               intermediate_size, traces,
               args.window_tokens, args.window_overlap,
               all_cats, done_keys, checkpoint_cb=_ckpt_cb_b)

    print(f"\nSerializing {len(all_cats)} categories …", flush=True)
    save = {
        "metadata": {
            "model": str(MODEL_PATH),
            "variant": f"v5-{args.variant}",
            "num_layers": num_layers,
            "num_experts": num_experts,
            "intermediate_size": intermediate_size,
            "hidden_size": hidden_size,
            "tier_a_max_tokens": args.tier_a_max_tokens,
            "tier_b_window_tokens": args.window_tokens,
            "tier_b_window_overlap": args.window_overlap,
            "tier_b_source": args.tier_b_json,
            "tier_b_trace_count": len(traces),
            "tier_b_set_counts": tier_b["metadata"]["set_counts"],
            "tier_b_set_weights": tier_b["metadata"]["set_weights"],
            "device": args.device,
            "dtype": args.dtype,
            "categories": list(all_cats.keys()),
        },
        "categories": serialize(all_cats, num_layers, num_experts),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(save, f)
    print(f"\nWrote {out_path}  ({out_path.stat().st_size/1024**2:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
