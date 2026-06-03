#!/usr/bin/env python3
"""T189 — generalized bucket router DIFFERENTIAL (128e teacher vs A2 prune).

Generalization of router_diff_constrained.py (T183) to an arbitrary loop bucket.
Phase 1: A2 generates greedy (rep_penalty=1.0) completions for <loop-bucket> +
openended prompts, records detect_loop. Phase 2: teacher-force those exact
sequences through full 128e, capture router softmax-over-128 on the COMPLETION
region, and with A2's keep-map accumulate per-(layer,expert) DROPPED mass on
loop tokens vs clean(openended) tokens.

Output: ranked dropped experts carrying <loop-bucket>-loop mass (HIGH loop / LOW
clean = the experts the drop map sacrificed). For --loop-bucket multilingual this
is the never-run clean analog of T183's constrained format-expert ID: it tells us
whether the multilingual loop capability is LOCALIZED (few high-specificity dropped
experts -> force-keep fixes it) or DIFFUSE (spread thin -> a 62e capacity limit).

Sequential single-GPU (A2 ~51G, then 128e ~49G; unload between). omk python.
"""
import argparse
import json
import time

import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_full_bench import detect_loop  # noqa: E402

A2 = os.environ.get("REDIST_STUDENT")
BASE = os.environ.get("REDIST_TEACHER")


def log(m):
    print("[diff %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2", default=A2, help="pruned student dir (or set REDIST_STUDENT)")
    ap.add_argument("--base", default=BASE, help="128e teacher dir (or set REDIST_TEACHER)")
    ap.add_argument("--keep-meta", default=os.environ.get("REDIST_KEEP_META"),
                    help="a2 keep metadata json (or set REDIST_KEEP_META)")
    ap.add_argument("--sample", default=os.environ.get("REDIST_SAMPLE"),
                    help="loop-screen prompts jsonl (or set REDIST_SAMPLE)")
    ap.add_argument("--loop-bucket", default="multilingual",
                    help="bucket whose LOOP tokens to attribute (vs openended clean)")
    ap.add_argument("--per-bucket", type=int, default=30)
    ap.add_argument("--gen-tokens", type=int, default=1024)
    ap.add_argument("--out", default="router_diff_bucket.json",
                    help="output json (CWD-relative default)")
    args = ap.parse_args()
    for _v, _fl, _ev in [(args.a2, "--a2", "REDIST_STUDENT"),
                         (args.base, "--base", "REDIST_TEACHER"),
                         (args.keep_meta, "--keep-meta", "REDIST_KEEP_META"),
                         (args.sample, "--sample", "REDIST_SAMPLE")]:
        if not _v:
            raise SystemExit(f"FAIL: {_fl} is required (or set {_ev})")
    LB = args.loop_bucket

    keep_meta = json.load(open(args.keep_meta))
    per_layer_keep = {int(k): sorted(v) for k, v in keep_meta["per_layer_keep"].items()}
    L = len(per_layer_keep)
    rows = [json.loads(x) for x in open(args.sample)]
    pools = {LB: [], "openended": []}
    for r in rows:
        b = r.get("bucket")
        if b in pools:
            pools[b].append(r["prompt"])
    work = ([(LB, p) for p in pools[LB][:args.per_bucket]]
            + [("openended", p) for p in pools["openended"][:args.per_bucket]])
    log("loop-bucket=%s  prompts: %s=%d openended=%d" % (
        LB, LB, min(len(pools[LB]), args.per_bucket), min(len(pools["openended"]), args.per_bucket)))

    tok = AutoTokenizer.from_pretrained(args.a2, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- Phase 1: A2 generates the (loop-prone) completions ----
    log("PHASE 1: load A2 -> cuda:0, generate greedy rep1.0")
    a2 = AutoModelForCausalLM.from_pretrained(
        args.a2, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": 0}).eval()
    seqs = []
    nlooped = 0
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
        if bucket == LB and looped:
            nlooped += 1
        seqs.append({"bucket": bucket, "looped": looped, "plen": plen, "ids": gen[0].tolist()})
    log("PHASE 1 done: %s loopers=%d / %d" % (LB, nlooped, min(len(pools[LB]), args.per_bucket)))
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
            cap[li] = out[0].detach().float().cpu()
        return hook
    handles = [routers[li][1].register_forward_hook(mk(li)) for li in range(L)]

    drop_mass_loop = np.zeros((L, E_full))
    drop_mass_clean = np.zeros((L, E_full))
    surv = {"loop": [], "loop_bucket_clean": [], "openended": []}
    n_loop_tok = 0
    n_clean_tok = 0
    for s in seqs:
        ids = torch.tensor([s["ids"]], device="cuda:0")
        cap.clear()
        with torch.no_grad():
            base(input_ids=ids, attention_mask=torch.ones_like(ids),
                 mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
        plen = s["plen"]
        is_loop = s["bucket"] == LB and s["looped"]
        is_oe = s["bucket"] == "openended"
        per_tok_surv = []
        for li in range(L):
            probs = cap[li]
            if probs.dim() == 3:
                probs = probs[0]
            probs = probs[plen:]
            keep = per_layer_keep[li]
            dropmask = np.ones(E_full, dtype=bool)
            dropmask[keep] = False
            per_tok_surv.append(probs[:, keep].sum(-1).numpy())
            col = probs.sum(0).numpy()
            if is_loop:
                drop_mass_loop[li] += col * dropmask
            elif is_oe:
                drop_mass_clean[li] += col * dropmask
        tok_surv = np.mean(np.stack(per_tok_surv, 0), axis=0)
        key = "loop" if is_loop else ("openended" if is_oe else "loop_bucket_clean")
        surv[key].extend(tok_surv.tolist())
        if is_loop:
            n_loop_tok += tok_surv.size
        elif is_oe:
            n_clean_tok += tok_surv.size
    for h in handles:
        h.remove()

    def stat(v):
        a = np.array(v)
        if not a.size:
            return {"n": 0}
        return {"n": int(a.size), "surv_median": round(float(np.median(a)), 4),
                "surv_mean": round(float(a.mean()), 4),
                "frac_below_0.3": round(float((a < 0.3).mean()), 4)}

    dl = drop_mass_loop / max(n_loop_tok, 1)
    dc = drop_mass_clean / max(n_clean_tok, 1)
    cands = []
    for li in range(L):
        for e in range(E_full):
            if dl[li, e] > 0.002:
                cands.append({"layer": li, "expert": e,
                              "loop_mass": round(float(dl[li, e]), 4),
                              "clean_mass": round(float(dc[li, e]), 4),
                              "specificity": round(float(dl[li, e] / (dc[li, e] + 1e-4)), 2)})
    cands.sort(key=lambda c: c["loop_mass"], reverse=True)

    out = {
        "loop_bucket": LB, "n_layers": L, "num_experts": E_full,
        "n_loopers": nlooped, "n_loop_tok": n_loop_tok, "n_clean_tok": n_clean_tok,
        "surviving_mass": {k: stat(v) for k, v in surv.items()},
        "total_dropped_mass_per_tok": {
            "loop": round(float(dl.sum()), 4), "clean": round(float(dc.sum()), 4)},
        "expert_candidates_top40": cands[:40],
        "n_candidates_over_thresh": len(cands),
    }
    json.dump(out, open(args.out, "w"), indent=1)
    log("=" * 64)
    log("LOOP BUCKET = %s   loopers=%d  loop_tok=%d  clean_tok=%d" % (LB, nlooped, n_loop_tok, n_clean_tok))
    for k, v in out["surviving_mass"].items():
        log("  %-20s n=%s surv_median=%s frac<0.3=%s" % (k, v.get("n"), v.get("surv_median"), v.get("frac_below_0.3")))
    log("  total dropped-mass/tok: loop=%.3f  clean=%.3f" % (
        out["total_dropped_mass_per_tok"]["loop"], out["total_dropped_mass_per_tok"]["clean"]))
    log("  top dropped-expert candidates (by %s-loop mass):" % LB)
    for c in cands[:15]:
        log("    L%-2d e%-3d  loop=%.3f clean=%.3f spec=%.1fx" % (
            c["layer"], c["expert"], c["loop_mass"], c["clean_mass"], c["specificity"]))
    log("DIFF_DONE -> %s" % args.out)


if __name__ == "__main__":
    main()
