#!/usr/bin/env python3
"""T176 — routing-collapse diagnostic: A2 (62e prune) vs 128e base on loop tokens.

WHY: 128e baseline = 0.0% loops on the same screen where A2 (62e+PES) loops 15.5%,
and loops are pruning-invariant to count/selection/magnitude (T175/T176). Working
hypothesis: pruning + router renormalization collapses the top-k routing
distribution on low-resource-language / format-constraint prompts into a
degenerate (low-entropy / top-1-saturated) state that sustains repetition, while
the unpruned base routes with healthy entropy on the SAME tokens.

TEST (apples-to-apples, content-controlled): generate A2's own greedy loop for a
set of looping prompts, then TEACHER-FORCE the identical prompt+loop token
sequence through BOTH models and compare, on the completion-region tokens:
  - normalized top-k routing entropy  H(softmax(router_logits)) / log(num_experts)
  - top-1 routing weight (saturation)
per layer, meaned. Pre-PES (PES rescales chosen experts post-top-k; it does not
change selection entropy). A2 on 62 experts, base on 128 — entropy normalized by
log(E) so the comparison is scale-fair.

VERDICT: if A2 normalized-entropy << base on the SAME loop tokens (and/or top-1
saturates), routing collapse is confirmed -> router-temperature/recalibration is
indicated. If A2 ~= base, the loop is not a simple routing-entropy collapse ->
SFT-heal is the better lever. Run on bs2 (base->cuda:1, A2->cuda:0), omk python.
"""
import argparse
import json
import math
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from audit_full_bench import detect_loop  # noqa: E402

BASE = "/srv/ml/models/base/gemma-4-26B-A4B-it"
A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"


def log(m):
    print("[probe %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def load(model_dir, dev):
    m = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": dev})
    m.eval()
    rh = {}
    routers = {n: mod for n, mod in m.named_modules() if n.endswith(".router")}
    assert len(routers) == 30, f"{model_dir}: {len(routers)} routers"

    def mk(name):
        def hook(_mod, args):
            rh[name] = args[0].detach()
        return hook
    hooks = [mod.register_forward_pre_hook(mk(n)) for n, mod in routers.items()]
    return m, routers, rh, hooks


@torch.no_grad()
def route_stats(model, routers, rh, ids, dev, start):
    """Mean (over layers) normalized top-k entropy + top-1 weight, per completion token.

    Returns (norm_entropy_mean, top1_mean) scalars over completion tokens [start:].
    """
    rh.clear()
    model(input_ids=ids, attention_mask=torch.ones_like(ids),
          mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
    ent_layers, top1_layers = [], []
    for name, mod in routers.items():
        h = rh.get(name)
        if h is None:
            continue
        h2 = h.reshape(-1, h.shape[-1])
        logits = F.linear(h2, mod.proj.weight).float()        # [T, E], pre-PES
        E = logits.shape[-1]
        p = logits.softmax(-1)
        ent = (-(p * (p.clamp_min(1e-12)).log()).sum(-1)) / math.log(E)
        top1 = p.max(-1).values
        ent_layers.append(ent[start:])
        top1_layers.append(top1[start:])
    ent = torch.stack(ent_layers).mean(0)     # per completion token, mean over layers
    top1 = torch.stack(top1_layers).mean(0)
    return ent.mean().item(), top1.mean().item(), ent.cpu().tolist(), top1.cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--pruned", default=A2)
    ap.add_argument("--sample", default="/mnt/sdc/ml/corpora/loop_screen_sample.jsonl")
    ap.add_argument("--buckets", default="multilingual,constrained")
    ap.add_argument("--per-bucket", type=int, default=12)
    ap.add_argument("--gen-tokens", type=int, default=1024)
    ap.add_argument("--out", default="/srv/ml/eval_results_tracks_2_3/t176_phase2/router_collapse.json")
    args = ap.parse_args()

    want = set(args.buckets.split(","))
    rows = [json.loads(x) for x in open(args.sample)]
    pools = {b: [r["prompt"] for r in rows if r.get("bucket") == b] for b in want}
    prompts = []
    for b in want:
        prompts += [(b, p) for p in pools[b][:args.per_bucket]]
    log("probing %d prompts (%s)" % (len(prompts), {b: min(len(pools[b]), args.per_bucket) for b in want}))

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    log("loading pruned (A2) -> cuda:0")
    pm, prout, prh, phk = load(args.pruned, 0)
    log("loading base (128e) -> cuda:1")
    bm, brout, brh, bhk = load(args.base, 1)

    results = []
    n_loop = 0
    for bucket, prompt in prompts:
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        penc = tok(chat, return_tensors="pt", add_special_tokens=False)
        plen = penc["input_ids"].shape[1]
        # 1) A2 greedy generation -> its loop
        with torch.no_grad():
            gen = pm.generate(**{k: v.to("cuda:0") for k, v in penc.items()},
                              max_new_tokens=args.gen_tokens, do_sample=False,
                              repetition_penalty=1.0, use_cache=True,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        comp_ids = gen[0][plen:]
        comp_txt = tok.decode(comp_ids, skip_special_tokens=True)
        looped = detect_loop(comp_txt)
        n_loop += int(looped)
        full = gen[0:1]                                   # prompt+completion
        # 2) teacher-force the SAME sequence through both, stats on completion region
        a_ent, a_top1, a_traj, _ = route_stats(pm, prout, prh, full.to("cuda:0"), 0, plen)
        b_ent, b_top1, b_traj, _ = route_stats(bm, brout, brh, full.to("cuda:1"), 0, plen)
        results.append({"bucket": bucket, "looped": bool(looped), "comp_len": int(comp_ids.numel()),
                        "a2_norm_entropy": a_ent, "base_norm_entropy": b_ent,
                        "a2_top1": a_top1, "base_top1": b_top1,
                        "prompt": prompt[:120]})
        log("%-12s loop=%-5s len=%4d | A2 H=%.3f top1=%.3f | base H=%.3f top1=%.3f | dH=%+.3f" % (
            bucket, looped, comp_ids.numel(), a_ent, a_top1, b_ent, b_top1, a_ent - b_ent))

    for h in phk + bhk:
        h.remove()

    # aggregate over LOOPING prompts (the population of interest)
    loops = [r for r in results if r["looped"]]
    pop = loops if loops else results
    import statistics as st
    agg = {
        "n_total": len(results), "n_loop": n_loop,
        "on": "looping_prompts" if loops else "all_prompts",
        "a2_norm_entropy_mean": round(st.mean(r["a2_norm_entropy"] for r in pop), 4),
        "base_norm_entropy_mean": round(st.mean(r["base_norm_entropy"] for r in pop), 4),
        "a2_top1_mean": round(st.mean(r["a2_top1"] for r in pop), 4),
        "base_top1_mean": round(st.mean(r["base_top1"] for r in pop), 4),
    }
    agg["entropy_gap_base_minus_a2"] = round(agg["base_norm_entropy_mean"] - agg["a2_norm_entropy_mean"], 4)
    json.dump({"agg": agg, "per_prompt": results}, open(args.out, "w"), indent=1)
    log("=" * 60)
    log("AGG over %s (n=%d, %d looped):" % (agg["on"], len(pop), n_loop))
    log("  A2   normalized-entropy=%.3f  top1=%.3f" % (agg["a2_norm_entropy_mean"], agg["a2_top1_mean"]))
    log("  base normalized-entropy=%.3f  top1=%.3f" % (agg["base_norm_entropy_mean"], agg["base_top1_mean"]))
    log("  base-minus-A2 entropy gap = %+.3f" % agg["entropy_gap_base_minus_a2"])
    if agg["entropy_gap_base_minus_a2"] > 0.05 or agg["a2_top1_mean"] - agg["base_top1_mean"] > 0.05:
        log("  VERDICT: ROUTING COLLAPSE in A2 on loop tokens -> router-temperature/recalibration indicated.")
    else:
        log("  VERDICT: no clear routing-entropy collapse -> SFT-heal is the better lever.")
    log("  -> %s" % args.out)


if __name__ == "__main__":
    main()
