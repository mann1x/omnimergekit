#!/usr/bin/env python3
"""probe_router_silentempty.py — T17.2 router probe.

Goal: diagnose why v4 produces silent-empty outputs on 7 problems (5 gsm8k +
HE/147 + HE+/107) where 128e succeeds.

For each (model, prompt) pair we capture, at the FINAL input token position:
  1. Top-20 logits over the vocabulary (next-token distribution)
  2. Per-layer router top-8 expert selection + probabilities (per token in last pos)
  3. Entropy of the (renormalized) router survivor distribution

Output: scripts/router_probe_silentempty.json
        + human-readable summary on stdout.

Runtime: CPU fp16, single forward pass per (model, prompt). ~30-60s per fwd ⇒
total ~15-25 min for 7 prompts × 2 models.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_TOKEN",
    open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    if Path("~/.cache/huggingface/token").expanduser().exists() else "")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch                                                            # noqa: E402

WS = Path("/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models")

MODELS = {
    "128e": WS / "google" / "gemma-4-26B-A4B-it",
    "v4":   WS / "google" / "gemma-4-A4B-98e-v4-it",
}

# Drop map used for v4 build — needed to identify which experts are absent in v4.
V4_DROP_MAP = WS / "scripts" / "cd_multiclass_98e_max_drop_map.json"


def _extract_user_content(arg0):
    """lm-eval arg_0 is a list whose [0] is a JSON-stringified [{role,content}]."""
    if isinstance(arg0, list) and arg0 and isinstance(arg0[0], str):
        try:
            msgs = json.loads(arg0[0])
        except json.JSONDecodeError:
            return arg0[0]
    elif isinstance(arg0, str):
        try:
            msgs = json.loads(arg0)
        except json.JSONDecodeError:
            return arg0
    else:
        msgs = arg0
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        return msgs[0].get("content", "")
    return ""


def load_prompts():
    """Return list of {bench, doc_id, prompt} for the 7 silent-empty cases."""
    out = []
    # gsm8k 1, 15, 42, 52, 99
    import glob
    fs = sorted(glob.glob(str(WS / "eval_results_vllm_suite/v4/gsm8k_100/98e_v4_nvfp4a16/lm_eval_out/98e_v4_nvfp4a16/samples_gsm8k_*.jsonl")))
    if fs:
        seen = set()
        with open(fs[-1]) as fh:
            for line in fh:
                d = json.loads(line)
                did = d["doc_id"]
                if did not in {1, 15, 42, 52, 99} or did in seen:
                    continue
                seen.add(did)
                arg0 = d.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "")
                out.append({"bench": "gsm8k", "doc_id": did,
                            "prompt": _extract_user_content(arg0)})
    # HE/147
    f = WS / "eval_results_vllm_suite/v4/humaneval_full/98e_v4_nvfp4a16/lm_eval_out/98e_v4_nvfp4a16/samples_humaneval_chat_2026-05-14T04-53-02.407959.jsonl"
    with open(f) as fh:
        for line in fh:
            d = json.loads(line)
            if d["doc_id"] != 147:
                continue
            arg0 = d.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "")
            out.append({"bench": "humaneval", "doc_id": 147,
                        "prompt": _extract_user_content(arg0)})
            break
    # HE+/107
    f = WS / "eval_results_vllm_suite/v4/humanevalplus_full/98e_v4_nvfp4a16/lm_eval_out/98e_v4_nvfp4a16/samples_humaneval_plus_chat_2026-05-14T06-40-41.916529.jsonl"
    with open(f) as fh:
        for line in fh:
            d = json.loads(line)
            if d["doc_id"] != 107:
                continue
            arg0 = d.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "")
            out.append({"bench": "humanevalplus", "doc_id": 107,
                        "prompt": _extract_user_content(arg0)})
            break
    return out


def install_router_hooks(model, num_layers):
    """Hook each MoE layer's experts module; record router top-8 + softmax at last pos."""
    records = {}
    hooks = []

    for li in range(num_layers):
        layer = model.model.language_model.layers[li]
        if not hasattr(layer, "experts"):
            continue

        def make_hook(layer_idx):
            def hook(module, args, output):
                # Gemma 4 flattens [B,T,D] → [B*T,D] BEFORE experts(), so:
                #   top_k_idx: [B*T, top_k]   top_k_wt: [B*T, top_k]
                # We probe with B=1, so last input position = last row.
                _, top_k_idx, top_k_wt = args
                last_t = top_k_idx.shape[0] - 1
                idx = top_k_idx[last_t].cpu().tolist()
                wt = top_k_wt[last_t].float().cpu().tolist()
                if not isinstance(idx, list):
                    idx = [idx]
                    wt = [wt]
                records[layer_idx] = {
                    "top_idx": idx,
                    "top_wt": wt,
                    "wt_sum": float(sum(wt)),
                    "num_experts": int(module.num_experts),
                }
            return hook
        hooks.append(layer.experts.register_forward_hook(make_hook(li)))

    return records, hooks


def run_one(model, tokenizer, prompt):
    """Single forward pass. Return next-token top-20 + router records."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True,
        add_generation_prompt=True, enable_thinking=True)
    input_ids = inputs["input_ids"]
    mm_ids = torch.zeros_like(input_ids)
    num_layers = model.config.text_config.num_hidden_layers

    records, hooks = install_router_hooks(model, num_layers)
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids, mm_token_type_ids=mm_ids)
    dt = time.time() - t0
    for h in hooks:
        h.remove()

    # Next-token logits = last position of out.logits
    last_logits = out.logits[0, -1].float()
    probs = torch.softmax(last_logits, dim=-1)
    top_p, top_i = torch.topk(probs, 20)
    top_logits = [{"token_id": int(i), "token": tokenizer.decode([int(i)]),
                   "prob": float(p), "logit": float(last_logits[int(i)])}
                  for i, p in zip(top_i.tolist(), top_p.tolist())]

    return {
        "input_tokens": int(input_ids.shape[1]),
        "elapsed_s": dt,
        "top20_next_token": top_logits,
        "router_per_layer": records,
    }


def diagnose_router_vs_baseline(v4_rec, base_rec, dropped_in_layer):
    """Compute: of the 8 experts 128e would have picked at last input position,
    how many are still alive in v4? What's the prob mass on dropped ones?"""
    base_top8 = set(base_rec["top_idx"])
    v4_alive = set(range(128)) - dropped_in_layer
    base_top8_alive = base_top8 & v4_alive
    base_top8_dropped = base_top8 - v4_alive

    # Approximate "lost prob mass": sum of base wt on dropped experts
    lost_mass = sum(w for i, w in zip(base_rec["top_idx"], base_rec["top_wt"])
                    if i in base_top8_dropped)
    return {
        "base_top8_kept_in_v4": len(base_top8_alive),
        "base_top8_dropped_in_v4": len(base_top8_dropped),
        "dropped_expert_ids_from_base_top8": sorted(base_top8_dropped),
        "approx_lost_prob_mass": float(lost_mass),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WS / "scripts" / "router_probe_silentempty.json"))
    ap.add_argument("--models", default="v4,128e",
                    help="Comma-sep. Default both (v4,128e). 128e load is ~5 min.")
    args = ap.parse_args()

    print("Loading prompts …", flush=True)
    prompts = load_prompts()
    print(f"  loaded {len(prompts)} prompts")
    for p in prompts:
        print(f"    {p['bench']}/{p['doc_id']}  prompt[:80]={p['prompt'][:80]!r}")

    # Load drop map (which experts are removed in v4 per layer)
    with open(V4_DROP_MAP) as f:
        dm = json.load(f)
    dropped = {int(k): set(v) for k, v in dm.items()}

    from transformers import AutoModelForCausalLM, AutoTokenizer

    requested = args.models.split(",")
    all_results = {"prompts": prompts, "drop_map_path": str(V4_DROP_MAP), "by_model": {}}

    for model_key in requested:
        model_dir = MODELS[model_key]
        print(f"\n=== Loading {model_key} from {model_dir} (CPU fp16) ===", flush=True)
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), dtype=torch.float16, device_map="cpu",
            trust_remote_code=True, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model.eval()
        print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

        by_prompt = {}
        for p in prompts:
            key = f"{p['bench']}/{p['doc_id']}"
            print(f"  [{model_key}] {key} …", flush=True)
            r = run_one(model, tokenizer, p["prompt"])
            print(f"    in_tok={r['input_tokens']} dt={r['elapsed_s']:.1f}s "
                  f"top1={r['top20_next_token'][0]['token']!r} "
                  f"p={r['top20_next_token'][0]['prob']:.3f}", flush=True)
            by_prompt[key] = r
        all_results["by_model"][model_key] = by_prompt

        del model, tokenizer
        gc.collect()

    # Cross-model diagnosis if both ran
    if "v4" in all_results["by_model"] and "128e" in all_results["by_model"]:
        diag = {}
        for key in all_results["by_model"]["128e"]:
            per_layer = {}
            for li_str, base_rec in all_results["by_model"]["128e"][key]["router_per_layer"].items():
                li = int(li_str)
                v4_rec = all_results["by_model"]["v4"][key]["router_per_layer"].get(li_str)
                if v4_rec is None:
                    continue
                per_layer[li] = diagnose_router_vs_baseline(v4_rec, base_rec, dropped.get(li, set()))
            diag[key] = per_layer
        all_results["diagnosis"] = diag

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: list(o) if isinstance(o, set) else o)
    print(f"\nwrote {out_path}")

    # Human-readable summary
    print("\n=== Summary: top next-token per prompt ===")
    print(f"{'prompt':<22} | {'model':<5} | {'top1':<20} {'p1':>5} | {'top2':<20} {'p2':>5}")
    for p in prompts:
        key = f"{p['bench']}/{p['doc_id']}"
        for m in requested:
            r = all_results["by_model"].get(m, {}).get(key)
            if not r:
                continue
            t1 = r["top20_next_token"][0]
            t2 = r["top20_next_token"][1]
            print(f"{key:<22} | {m:<5} | {t1['token']!r:<20} {t1['prob']:>5.2f} | "
                  f"{t2['token']!r:<20} {t2['prob']:>5.2f}")

    if "diagnosis" in all_results:
        print("\n=== Per-prompt: base-top8 dropped-in-v4 (summed across 30 layers) ===")
        for key, per_layer in all_results["diagnosis"].items():
            tot_dropped = sum(d["base_top8_dropped_in_v4"] for d in per_layer.values())
            tot_lost = sum(d["approx_lost_prob_mass"] for d in per_layer.values())
            print(f"  {key:<22}: {tot_dropped:>3} of {len(per_layer)*8} 128e-top8 slots dropped; "
                  f"sum lost prob mass ≈ {tot_lost:.2f}")


if __name__ == "__main__":
    main()
