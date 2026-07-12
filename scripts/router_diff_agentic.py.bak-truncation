#!/usr/bin/env python3
"""Agentic clean-vs-loop router DIFFERENTIAL (128e teacher vs fkbroad selection).

Broadened successor to T183 router_diff_constrained.py and the NARROW agentic_eog
emit-position map. The T202.3 PRE-CHECK showed terminator-EMIT experts are ~97%
already kept by fkbroad (narrow signal exhausted). This windows over the WHOLE
agentic trajectory instead of the single emit token:

  Phase 1  fkbroad-SELECTION (the no-fold student = expert_drop(fkbroad) + shared
           a=1.2, NO DERN fold) generates on the agentic fixtures under the vendor
           _minp loop sampler (t0.9 top_p.95 top_k64 min_p.05 rep1.1). detect_loop
           + hit-cap label each seed loop/clean. (fkbroad loops here; 128e does NOT
           -> 0/48 on these fixtures, so the loop is purely prune-induced.)
  Phase 2  the 128e teacher teacher-forces EVERY sequence, captures router softmax
           over 128, and with fkbroad's drop map accumulates per-(layer,expert)
           DROPPED mass over loop-region tokens vs clean-region tokens.

Decision:
  total dropped-mass/tok loop >> clean  AND a consistent specific dropped set
        -> loop-relevant experts the SELECTION sacrificed == additive force-keep
           targets (feed top pins to gen_drop_v5_fk.py --force-keep).
  dropped-mass loop ~= clean (both low)
        -> the loop is NOT a dropped-expert problem (kept-expert sharpness / the
           average-fold blur owns the 0/48) -> force-keep cannot help; confirms
           the PRE-CHECK and the no-fold-gate premise.

Keep/drop are in ORIGINAL 128e numbering (the drop map lists original ids); the
looper's 98e renumbering is irrelevant -- it is only used to GENERATE token text.
Sequential single-GPU: looper (~40G, sdpa) Phase 1, unload, 128e (~49G, eager)
Phase 2. Run on bs2 GPU1, omk python. New file -- does not touch the working
router_diff_constrained.py.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from audit_full_bench import detect_loop  # noqa: E402


def log(m):
    print("[adiff %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--looper", required=True)  # fkbroad selection bf16 (no-fold student)
    ap.add_argument("--base", default="/srv/ml/models/base/gemma-4-26B-A4B-it")
    ap.add_argument("--drop-map", default="/srv/ml/scripts/v8coder_fkbroad_drop_map.json")
    ap.add_argument("--fixtures-dir", default="/srv/ml/agentic_loop/fixtures")
    ap.add_argument("--fixtures",
                    default="solar_build_start,threejs_build")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--gen-tokens", type=int, default=2048)
    # flash-attn is absent + fixtures are 20k-87k tokens -> SDPA math materializes
    # [1,~30,T,T] and OOMs. The loop is a LOCAL repetition, so the routing signal
    # lives in the completion region, not the full history. Tail-truncate the prompt
    # to a memory-feasible window (30*N^2*2 bytes transient). 0 = no truncation.
    ap.add_argument("--max-prompt-tokens", type=int, default=6144)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--rep", type=float, default=1.1)
    ap.add_argument("--out", default="/mnt/sdc/ml/google/expert_neuron_v8_agentic_diff.json")
    args = ap.parse_args()

    drop = json.load(open(args.drop_map))
    dropmap = {int(k): set(int(e) for e in v) for k, v in drop.items()}
    log("drop map: %d layers, %d dropped/layer" % (len(dropmap), len(next(iter(dropmap.values())))))

    fixtures = []
    for nm in args.fixtures.split(","):
        p = os.path.join(args.fixtures_dir, nm + ".json")
        if os.path.exists(p):
            fixtures.append((nm, json.load(open(p))))
        else:
            log("WARN fixture missing: %s" % p)
    log("fixtures: %s" % ",".join(n for n, _ in fixtures))

    tok = AutoTokenizer.from_pretrained(args.looper, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- Phase 1: fkbroad looper generates (sampled vendor_minp), label loop/clean ----
    log("PHASE 1: load looper -> cuda:0 (sdpa, sampled vendor_minp t=%.2f minp=%.2f rep=%.2f)"
        % (args.temp, args.min_p, args.rep))
    looper = AutoModelForCausalLM.from_pretrained(
        args.looper, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="sdpa", device_map={"": 0}).eval()
    seqs = []
    for nm, fx in fixtures:
        msgs = fx["messages"]
        tools = fx.get("tools")
        try:
            chat = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                           tokenize=False)
        except Exception as e:  # noqa: BLE001
            log("  (%s) tools template failed (%s); falling back to messages-only" % (nm, e))
            chat = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(chat, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        full_plen = enc["input_ids"].shape[1]
        if args.max_prompt_tokens > 0 and full_plen > args.max_prompt_tokens:
            # keep the TAIL (recent context + generation prompt) so the agentic
            # continuation framing + loop-prone recent tokens survive
            enc = {k: v[:, -args.max_prompt_tokens:] for k, v in enc.items()}
            log("  (%s) prompt %d -> tail %d tokens" % (nm, full_plen, enc["input_ids"].shape[1]))
        plen = enc["input_ids"].shape[1]
        for s in range(args.seeds):
            torch.manual_seed(args.seed_base + s)
            with torch.no_grad():
                gen = looper.generate(
                    **enc, max_new_tokens=args.gen_tokens, do_sample=True,
                    temperature=args.temp, top_p=args.top_p, top_k=args.top_k,
                    min_p=args.min_p, repetition_penalty=args.rep, use_cache=True,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
            comp = gen[0][plen:]
            text = tok.decode(comp, skip_special_tokens=True)
            hit_cap = comp.numel() >= args.gen_tokens - 2
            looped = bool(detect_loop(text)) or hit_cap
            seqs.append({"fixture": nm, "seed": args.seed_base + s, "looped": looped,
                         "hit_cap": bool(hit_cap), "plen": int(plen),
                         "comp_len": int(comp.numel()), "ids": gen[0].tolist()})
        nl = sum(1 for x in seqs if x["fixture"] == nm and x["looped"])
        log("  %-22s loops=%d/%d" % (nm, nl, args.seeds))
    del looper
    torch.cuda.empty_cache()
    n_loop = sum(1 for s in seqs if s["looped"])
    n_clean = len(seqs) - n_loop
    log("PHASE 1 done: %d seqs (%d loop / %d clean)" % (len(seqs), n_loop, n_clean))
    if n_loop == 0 or n_clean == 0:
        log("WARN one-sided split loop=%d clean=%d -- differential weak; widen seeds/fixtures"
            % (n_loop, n_clean))

    # ---- Phase 2: teacher-force through full 128e, capture router softmax ----
    log("PHASE 2: load 128e -> cuda:0 (sdpa), teacher-force + capture routing")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="sdpa", device_map={"": 0}).eval()
    E_full = base.config.text_config.num_experts
    routers = sorted([(int(n.split("layers.")[1].split(".")[0]), m)
                      for n, m in base.named_modules() if n.endswith(".router")],
                     key=lambda t: t[0])
    Lr = len(routers)
    log("  routers=%d  num_experts=%d" % (Lr, E_full))
    cap = {}

    def mk(li):
        def hook(_m, _i, out):
            cap[li] = out[0].detach().float().cpu()  # router_probabilities [T,128]
        return hook

    handles = [routers[li][1].register_forward_hook(mk(li)) for li in range(Lr)]

    drop_mass_loop = np.zeros((Lr, E_full))
    drop_mass_clean = np.zeros((Lr, E_full))
    surv = {"loop": [], "clean": []}
    n_loop_tok = 0
    n_clean_tok = 0
    for s in seqs:
        ids = torch.tensor([s["ids"]], device="cuda:0")
        cap.clear()
        with torch.no_grad():
            base(input_ids=ids, attention_mask=torch.ones_like(ids),
                 mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
        plen = s["plen"]
        is_loop = s["looped"]
        per_tok_surv = []
        for li in range(Lr):
            probs = cap[li]
            if probs.dim() == 3:
                probs = probs[0]
            probs = probs[plen:]  # completion region [T,128]
            if probs.shape[0] == 0:
                continue
            dropmask = np.zeros(E_full, dtype=bool)
            for e in dropmap.get(li, set()):
                if 0 <= e < E_full:
                    dropmask[e] = True
            keepmask = ~dropmask
            per_tok_surv.append(probs[:, keepmask].sum(-1).numpy())
            col = probs.sum(0).numpy()
            if is_loop:
                drop_mass_loop[li] += col * dropmask
            else:
                drop_mass_clean[li] += col * dropmask
        if per_tok_surv:
            m = min(a.size for a in per_tok_surv)
            tok_surv = np.mean(np.stack([a[:m] for a in per_tok_surv], 0), axis=0)
            surv["loop" if is_loop else "clean"].extend(tok_surv.tolist())
            if is_loop:
                n_loop_tok += tok_surv.size
            else:
                n_clean_tok += tok_surv.size
    for h in handles:
        h.remove()

    dlm = drop_mass_loop / max(n_loop_tok, 1)
    dcm = drop_mass_clean / max(n_clean_tok, 1)
    cands = []
    for li in range(Lr):
        for e in sorted(dropmap.get(li, set())):
            if 0 <= e < E_full and dlm[li, e] > 0.001:
                cands.append({"layer": li, "expert": int(e),
                              "loop_mass": round(float(dlm[li, e]), 4),
                              "clean_mass": round(float(dcm[li, e]), 4),
                              "specificity": round(float(dlm[li, e] / (dcm[li, e] + 1e-4)), 2)})
    cands.sort(key=lambda c: c["loop_mass"], reverse=True)

    def stat(v):
        a = np.array(v)
        return ({"n": int(a.size),
                 "surv_median": round(float(np.median(a)), 4),
                 "surv_mean": round(float(a.mean()), 4)} if a.size else {"n": 0})

    out = {
        "n_layers": Lr, "num_experts": E_full,
        "n_loop_seqs": n_loop, "n_clean_seqs": n_clean,
        "surviving_mass": {k: stat(v) for k, v in surv.items()},
        "total_dropped_mass_per_tok": {
            "loop": round(float(dlm.sum()), 4),
            "clean": round(float(dcm.sum()), 4)},
        "dropped_loop_expert_candidates": cands[:40],
        "n_candidates_over_thresh": len(cands),
        "params": {"fixtures": [n for n, _ in fixtures], "seeds": args.seeds,
                   "temp": args.temp, "min_p": args.min_p, "rep": args.rep,
                   "gen_tokens": args.gen_tokens},
    }
    json.dump(out, open(args.out, "w"), indent=1)
    log("=" * 64)
    log("seqs: %d loop / %d clean   (loop_tok=%d clean_tok=%d)" % (
        n_loop, n_clean, n_loop_tok, n_clean_tok))
    for k, v in out["surviving_mass"].items():
        log("  surv_mass %-6s n=%s median=%s mean=%s" % (
            k, v.get("n"), v.get("surv_median"), v.get("surv_mean")))
    tl = out["total_dropped_mass_per_tok"]["loop"]
    tc = out["total_dropped_mass_per_tok"]["clean"]
    log("  total dropped-mass/tok: loop=%.3f clean=%.3f  (ratio=%.2fx)" % (
        tl, tc, tl / max(tc, 1e-6)))
    log("  top dropped loop-expert candidates (by loop_mass):")
    for c in cands[:12]:
        log("    L%-2d e%-3d  loop=%.4f clean=%.4f spec=%.1fx" % (
            c["layer"], c["expert"], c["loop_mass"], c["clean_mass"], c["specificity"]))
    log("ADIFF_DONE -> %s" % args.out)


if __name__ == "__main__":
    main()
