#!/usr/bin/env python3
"""REDIST localize — fluent-failure divergence signal (T191 Phase-0 unblock).

The loop-differential (router_diff_bucket.py) localizes a capability by attributing
the teacher's DROPPED-expert routed mass to LOOP tokens. That only works for
capabilities that fail by *looping* (multilingual, constrained-format). Capabilities
like CODE / SCIENCE / MATH fail *fluently* — wrong-but-coherent, no loops — so
detect_loop fires on nothing and the loop-differential is blind.

This script is the fluent analog, grader-free:

  Phase 1  the pruned STUDENT (A2) generates greedy completions on the driver corpus.
  Phase 2  load teacher(128e) + student RESIDENT on two GPUs; for each sequence,
           forward BOTH, capture the teacher's router softmax-over-128 (hooks) and
           compute per-completion-token next-token KL( teacher || student ).
           Tokens where the student's distribution DIVERGES from the teacher are
           exactly where the prune removed capability. Accumulate the teacher's
           dropped-expert routed mass on the top-KL ("divergent") tokens vs the
           bottom-KL ("agree") tokens.

Output (schema matches router_diff_bucket.py so `redist.py localize` is uniform):
  expert_candidates_top40 : dropped experts carrying divergence mass, by specificity
  n_candidates_over_thresh, concentration (top-16 share), shape {localized|diffuse}

Two-GPU resident (teacher cuda:0 ~50G, student cuda:1 ~50G) — fits bs2 Blackwell.
omk python. Mirrors router_diff_bucket.py hook/IO style.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

A2 = os.environ.get("REDIST_STUDENT")
BASE = os.environ.get("REDIST_TEACHER")


def log(m):
    print("[divloc %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def _load(path, device):
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager",
        device_map={"": device}).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=A2, help="pruned student dir (or set REDIST_STUDENT)")
    ap.add_argument("--teacher", default=BASE, help="128e teacher dir (or set REDIST_TEACHER)")
    ap.add_argument("--keep-meta", default=os.environ.get("REDIST_KEEP_META"),
                    help="a2 keep metadata json (or set REDIST_KEEP_META)")
    ap.add_argument("--corpus", required=True,
                    help="jsonl with {prompt} (or {prompt,bucket}); the driver capability")
    ap.add_argument("--per-n", type=int, default=40, help="num prompts to use")
    ap.add_argument("--gen-tokens", type=int, default=1024)
    ap.add_argument("--kl-top-frac", type=float, default=0.20,
                    help="fraction of highest-KL completion tokens treated as 'divergent'")
    ap.add_argument("--teacher-device", default="cuda:0")
    ap.add_argument("--student-device", default="cuda:1")
    ap.add_argument("--out", default="localize_fluent.json",
                    help="output json (CWD-relative default)")
    args = ap.parse_args()
    for _v, _fl, _ev in [(args.student, "--student", "REDIST_STUDENT"),
                         (args.teacher, "--teacher", "REDIST_TEACHER"),
                         (args.keep_meta, "--keep-meta", "REDIST_KEEP_META")]:
        if not _v:
            raise SystemExit(f"FAIL: {_fl} is required (or set {_ev})")

    keep_meta = json.load(open(args.keep_meta))
    per_layer_keep = {int(k): sorted(v) for k, v in keep_meta["per_layer_keep"].items()}
    L = len(per_layer_keep)

    rows = [json.loads(x) for x in open(args.corpus)]
    prompts = [r["prompt"] for r in rows][:args.per_n]
    log("corpus=%s prompts=%d gen_tokens=%d kl_top_frac=%.2f" % (
        args.corpus, len(prompts), args.gen_tokens, args.kl_top_frac))

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- Phase 1: student generates greedy completions ----
    log("PHASE 1: load student -> %s, generate greedy rep1.0" % args.student_device)
    student = _load(args.student, args.student_device)
    sd = args.student_device
    seqs = []
    for p in prompts:
        chat = tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)
        enc = tok(chat, return_tensors="pt", add_special_tokens=False).to(sd)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = student.generate(**enc, max_new_tokens=args.gen_tokens, do_sample=False,
                                   repetition_penalty=1.0, use_cache=True,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
        seqs.append({"plen": plen, "ids": gen[0].tolist()})
    log("PHASE 1 done: %d sequences" % len(seqs))

    # ---- Phase 2: teacher + student RESIDENT; per-token KL + teacher dropped-mass ----
    log("PHASE 2: load teacher -> %s (student stays on %s)" % (args.teacher_device, sd))
    teacher = _load(args.teacher, args.teacher_device)
    td = args.teacher_device
    E_full = teacher.config.text_config.num_experts

    cap = {}
    routers = sorted([(int(n.split("layers.")[1].split(".")[0]), m)
                      for n, m in teacher.named_modules() if n.endswith(".router")],
                     key=lambda t: t[0])

    def mk(li):
        def hook(_m, _i, out):
            cap[li] = out[0].detach().float().cpu()
        return hook
    handles = [routers[li][1].register_forward_hook(mk(li)) for li in range(L)]

    drop_mass_div = np.zeros((L, E_full))   # dropped-mass on divergent tokens
    drop_mass_agr = np.zeros((L, E_full))   # dropped-mass on agree tokens
    n_div_tok = 0
    n_agr_tok = 0
    for s in seqs:
        plen = s["plen"]
        ids_t = torch.tensor([s["ids"]], device=td)
        ids_s = torch.tensor([s["ids"]], device=sd)
        cap.clear()
        with torch.no_grad():
            ot = teacher(input_ids=ids_t, attention_mask=torch.ones_like(ids_t),
                         mm_token_type_ids=torch.zeros_like(ids_t), use_cache=False)
            os_ = student(input_ids=ids_s, attention_mask=torch.ones_like(ids_s),
                          mm_token_type_ids=torch.zeros_like(ids_s), use_cache=False)
        # next-token KL(teacher||student) on completion region, computed on teacher dev
        lt = ot.logits[0, plen - 1:-1].float()        # [Tc, V] on td
        ls = os_.logits[0, plen - 1:-1].float().to(td)
        logp_t = torch.log_softmax(lt, -1)
        logp_s = torch.log_softmax(ls, -1)
        p_t = logp_t.exp()
        kl = (p_t * (logp_t - logp_s)).sum(-1).cpu().numpy()   # [Tc]
        del lt, ls, logp_t, logp_s, p_t, ot, os_
        if kl.size < 4:
            continue
        thr = np.quantile(kl, 1.0 - args.kl_top_frac)
        div_mask = kl >= thr     # divergent (high-KL) completion tokens
        agr_mask = kl <= np.quantile(kl, args.kl_top_frac)
        for li in range(L):
            probs = cap[li]
            if probs.dim() == 3:
                probs = probs[0]
            probs = probs[plen:]                 # completion region, [Tc(+1), E]
            probs = probs[:kl.shape[0]]          # align to KL length
            keep = per_layer_keep[li]
            dropmask = np.ones(E_full, dtype=bool)
            dropmask[keep] = False
            pv = probs.numpy()
            drop_mass_div[li] += (pv[div_mask].sum(0)) * dropmask
            drop_mass_agr[li] += (pv[agr_mask].sum(0)) * dropmask
        n_div_tok += int(div_mask.sum())
        n_agr_tok += int(agr_mask.sum())
    for h in handles:
        h.remove()

    dd = drop_mass_div / max(n_div_tok, 1)
    da = drop_mass_agr / max(n_agr_tok, 1)
    cands = []
    for li in range(L):
        for e in range(E_full):
            if dd[li, e] > 0.002:
                cands.append({"layer": li, "expert": e,
                              "div_mass": round(float(dd[li, e]), 4),
                              "agree_mass": round(float(da[li, e]), 4),
                              "specificity": round(float(dd[li, e] / (da[li, e] + 1e-4)), 2)})
    cands.sort(key=lambda c: c["div_mass"], reverse=True)

    total_div = float(dd.sum())
    top16 = sum(c["div_mass"] for c in cands[:16])
    concentration = round(top16 / (total_div + 1e-9), 4)
    shape = "localized" if concentration >= 0.40 else "diffuse"

    out = {
        "signal": "fluent_divergence",
        "driver_corpus": args.corpus,
        "n_layers": L, "num_experts": E_full,
        "n_div_tok": n_div_tok, "n_agree_tok": n_agr_tok,
        "total_div_dropped_mass_per_tok": round(total_div, 4),
        "concentration_top16": concentration,
        "shape": shape,
        "expert_candidates_top40": cands[:40],
        "n_candidates_over_thresh": len(cands),
    }
    json.dump(out, open(args.out, "w"), indent=1)
    log("=" * 64)
    log("FLUENT DIVERGENCE  div_tok=%d agree_tok=%d  total_div_mass=%.3f" % (
        n_div_tok, n_agr_tok, total_div))
    log("concentration(top16)=%.3f -> shape=%s  (n_cands=%d)" % (
        concentration, shape, len(cands)))
    for c in cands[:15]:
        log("  L%-2d e%-3d  div=%.3f agree=%.3f spec=%.1fx" % (
            c["layer"], c["expert"], c["div_mass"], c["agree_mass"], c["specificity"]))
    log("DIVLOC_DONE -> %s" % args.out)


if __name__ == "__main__":
    main()
