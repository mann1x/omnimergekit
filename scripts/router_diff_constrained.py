#!/usr/bin/env python3
"""T183 constrained-bucket router DIFFERENTIAL (128e teacher vs A2 prune).

Constrained-format loops are rep-penalty-IMMUNE (flat ~7.5% across rp1.0/1.05/1.1,
T182) -> a different mechanism than the multilingual repetition attractor. KEY
ARCHITECTURAL FACT: expert_drop.py slices router.proj ROWS to survivors, so A2's
ranking of its 62 survivors == 128e's ranking of those same 62 (monotonic softmax).
A2 already routes "like 128e among survivors"; the only prune-induced change is
(i) 128e's dropped top-8 experts are unreachable and (ii) the softmax denominator
shrank. So the decisive question for whether ANY router-math fix exists for the
constrained loop is: did the constrained-format capability live in DROPPED or
SURVIVING experts? T180 measured this for multilingual/openended on PROMPT tokens
only -- never constrained, never on loop completions.

This measures it directly. Phase 1: A2 generates greedy (rep_penalty=1.0, the loop-
prone config) constrained completions, record token ids + detect_loop. Phase 2:
teacher-force those exact sequences through the FULL 128e, capture router
softmax-over-128, and on the COMPLETION region compute, with the A2 keep-map:
  - surviving-mass = sum softmax_128[keep] : constrained-loop vs constrained-nonloop
    vs clean(openended). HIGH dropped-mass on constrained-loop -> capability pruned
    (same wall as multilingual; only a re-pruned drop map helps). LOW -> format
    experts survived, routing isn't the cause, a Delta-bias calibration could help.
  - per-(layer,expert) DROPPED mass accumulated over constrained-loop tokens vs over
    clean tokens -> ranks the specific dropped experts carrying constrained-format
    mass. A consistent set with HIGH constrained / LOW clean mass == "format experts"
    the drop map sacrificed -> the exact experts to protect in a re-prune.

Sequential single-GPU (A2 ~51G, then 128e ~49G; unload between) -> fits GPU0 alone.
Run on bs2 GPU0, omk python.
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from audit_full_bench import detect_loop  # noqa: E402

A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"
BASE = "/srv/ml/models/base/gemma-4-26B-A4B-it"


def log(m):
    print("[diff %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2", default=A2)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--keep-meta", default="/srv/ml/scripts/a2_keep_metadata.json")
    ap.add_argument("--sample", default="/mnt/sdc/ml/corpora/loop_screen_sample.jsonl")
    ap.add_argument("--per-bucket", type=int, default=20)
    ap.add_argument("--gen-tokens", type=int, default=1024)
    ap.add_argument("--out", default="/srv/ml/eval_results_tracks_2_3/t176_phase3/router_diff_constrained.json")
    args = ap.parse_args()

    keep_meta = json.load(open(args.keep_meta))
    per_layer_keep = {int(k): sorted(v) for k, v in keep_meta["per_layer_keep"].items()}
    L = len(per_layer_keep)
    rows = [json.loads(x) for x in open(args.sample)]
    pools = {"constrained": [], "openended": []}
    for r in rows:
        b = r.get("bucket")
        if b in pools:
            pools[b].append(r["prompt"])
    work = ([("constrained", p) for p in pools["constrained"][:args.per_bucket]]
            + [("openended", p) for p in pools["openended"][:args.per_bucket]])
    log("prompts: constrained=%d openended=%d" % (
        min(len(pools["constrained"]), args.per_bucket), min(len(pools["openended"]), args.per_bucket)))

    tok = AutoTokenizer.from_pretrained(args.a2, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- Phase 1: A2 generates the (loop-prone) completions ----
    log("PHASE 1: load A2 -> cuda:0, generate greedy rep1.0")
    a2 = AutoModelForCausalLM.from_pretrained(
        args.a2, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": 0}).eval()
    seqs = []
    for bucket, prompt in work:
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        enc = tok(chat, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = a2.generate(**enc, max_new_tokens=args.gen_tokens, do_sample=False,
                              repetition_penalty=1.0, use_cache=True,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        comp = gen[0][plen:]
        looped = bool(detect_loop(tok.decode(comp, skip_special_tokens=True)))
        seqs.append({"bucket": bucket, "looped": looped, "plen": plen,
                     "ids": gen[0].tolist()})
        log("  %-12s loop=%-5s comp_len=%d" % (bucket, looped, comp.numel()))
    del a2
    torch.cuda.empty_cache()

    # ---- Phase 2: teacher-force through full 128e, capture router softmax ----
    log("PHASE 2: load 128e -> cuda:0, teacher-force + capture routing")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": 0}).eval()
    cap = {}
    routers = sorted([(int(n.split("layers.")[1].split(".")[0]), m)
                      for n, m in base.named_modules() if n.endswith(".router")], key=lambda t: t[0])
    E_full = base.config.text_config.num_experts

    def mk(li):
        def hook(_m, _i, out):
            cap[li] = out[0].detach().float().cpu()  # router_probabilities [T,128]
        return hook
    handles = [routers[li][1].register_forward_hook(mk(li)) for li in range(L)]

    # per-(layer,expert) dropped-mass accumulators
    drop_mass_loop = np.zeros((L, E_full))
    drop_mass_clean = np.zeros((L, E_full))
    surv = {"constrained_loop": [], "constrained_clean": [], "openended": []}
    n_loop_tok = 0
    n_clean_tok = 0
    for s in seqs:
        ids = torch.tensor([s["ids"]], device="cuda:0")
        cap.clear()
        with torch.no_grad():
            base(input_ids=ids, attention_mask=torch.ones_like(ids),
                 mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
        plen = s["plen"]
        is_loop = s["bucket"] == "constrained" and s["looped"]
        is_oe = s["bucket"] == "openended"
        per_tok_surv = []
        for li in range(L):
            probs = cap[li]
            if probs.dim() == 3:
                probs = probs[0]
            probs = probs[plen:]                       # completion region [T,128]
            keep = per_layer_keep[li]
            dropmask = np.ones(E_full, dtype=bool)
            dropmask[keep] = False
            per_tok_surv.append(probs[:, keep].sum(-1).numpy())
            col = probs.sum(0).numpy()                 # mass per expert summed over tokens
            if is_loop:
                drop_mass_loop[li] += col * dropmask
            elif is_oe:
                drop_mass_clean[li] += col * dropmask
        tok_surv = np.mean(np.stack(per_tok_surv, 0), axis=0)  # [T] layer-avg
        key = "constrained_loop" if is_loop else ("openended" if is_oe else "constrained_clean")
        surv[key].extend(tok_surv.tolist())
        if is_loop:
            n_loop_tok += tok_surv.size
        elif is_oe:
            n_clean_tok += tok_surv.size
    for h in handles:
        h.remove()

    def stat(v):
        a = np.array(v)
        return {"n": int(a.size), "surv_median": round(float(np.median(a)), 4) if a.size else None,
                "surv_mean": round(float(a.mean()), 4) if a.size else None,
                "frac_below_0.3": round(float((a < 0.3).mean()), 4) if a.size else None} if a.size else {"n": 0}

    # normalize per-token then rank format-expert candidates
    dl = drop_mass_loop / max(n_loop_tok, 1)
    dc = drop_mass_clean / max(n_clean_tok, 1)
    cands = []
    for li in range(L):
        for e in range(E_full):
            if dl[li, e] > 0.002:                      # meaningful per-token dropped mass on loops
                cands.append({"layer": li, "expert": e,
                              "constrained_loop_mass": round(float(dl[li, e]), 4),
                              "clean_mass": round(float(dc[li, e]), 4),
                              "specificity": round(float(dl[li, e] / (dc[li, e] + 1e-4)), 2)})
    cands.sort(key=lambda c: c["constrained_loop_mass"], reverse=True)

    out = {
        "n_layers": L, "num_experts": E_full,
        "surviving_mass": {k: stat(v) for k, v in surv.items()},
        "total_dropped_mass_per_tok": {
            "constrained_loop": round(float(dl.sum()), 4),
            "openended_clean": round(float(dc.sum()), 4)},
        "format_expert_candidates_top30": cands[:30],
        "n_candidates_over_thresh": len(cands),
    }
    json.dump(out, open(args.out, "w"), indent=1)
    log("=" * 64)
    for k, v in out["surviving_mass"].items():
        log("  %-20s n=%s surv_median=%s frac<0.3=%s" % (k, v.get("n"), v.get("surv_median"), v.get("frac_below_0.3")))
    log("  total dropped-mass/tok: constrained_loop=%.3f  clean=%.3f" % (
        out["total_dropped_mass_per_tok"]["constrained_loop"],
        out["total_dropped_mass_per_tok"]["openended_clean"]))
    log("  top format-expert candidates (dropped, by constrained_loop_mass):")
    for c in cands[:8]:
        log("    L%-2d e%-3d  loop=%.3f clean=%.3f spec=%.1fx" % (
            c["layer"], c["expert"], c["constrained_loop_mass"], c["clean_mass"], c["specificity"]))
    log("DIFF_DONE -> %s" % args.out)


if __name__ == "__main__":
    main()
