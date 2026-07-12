#!/usr/bin/env python3
"""redist_dern_eq11.py — T203: survivor-anchored DERN Eq.11 norm-equalized
redistribution of dropped experts INTO the 98e v7-coder survivors, TRUE SEQUENTIAL
(downstream-aware). User-directed (2026-06-17).

Motivation (T197): the v7-coder agentic loop residual vs the 128e teacher is
DIRECTIONAL drift (cos~0.83 mid-stack), magnitude already matched. Pure HC-SMoE
freq-weighted averaging would COLLAPSE the (correct) magnitude; the missing step is
DERN Eq.11 norm-equalization. We anchor it on the SURVIVOR: kept expert i is the
sole reference for routing identity (its router row is untouched) AND the norm
target. Dropped experts only fold IN — never the reverse — so nothing in the result
"points to" a pruned expert.

Downstream-routing chased STRONGLY via single-pass SEQUENTIAL merge: each layer's
fold is fit on the hidden state the *merged-so-far* student actually produces, so
the corrected trajectory stays in the survivor-served region at every later layer.

Pipeline (one process, never both full models resident):
  Phase A  load 128e teacher -> forward agentic calib -> freq[L,128] -> free teacher
  Phase B  load 98e student  -> ONE forward of a long agentic transcript:
             per-layer .experts pre-hook: x_L (live, merged-upstream) ->
               stream teacher experts[L] from disk ->
               assign dropped -> nearest SURVIVING teacher expert (cosine) ->
               fold = freq-weighted avg of (survivor + assigned dropped) ->
               Eq.11 survivor-anchor: scale merged down_proj so per-token output
                 norm == survivor's original output norm on x_L ->
               overwrite student .experts.{gate_up_proj,down_proj} IN PLACE
           -> save_pretrained(merged student)

Gate downstream AR-first (loop_screen / 12-seed agentic gate), NEVER recon-MSE.

Usage:
  redist_dern_eq11.py --teacher <128e> --student <98e> --keep-meta <json>
      --freq-corpus <jsonl> --seq-corpus <jsonl> --out <dir>
      [--freq-max-seqs 64 --freq-max-tokens 4096 --seq-max-tokens 8192 --device cuda:0]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# reuse validated primitives from the omk redist framework
sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from redist import _assign_dropped, _load_tensor, _st_index  # noqa: E402

EXP = "model.language_model.layers.%d.experts.%s"


def log(m):
    print(m, flush=True)


def _act(x):
    return F.gelu(x, approximate="tanh")  # gelu_pytorch_tanh


def _expert_out(x, gate_up, down):
    """SwiGLU output of ONE expert. x:[T,H] f32, gate_up:[2M,H], down:[H,M]. -> [T,H]."""
    h = x @ gate_up.T
    M = down.shape[-1]
    g, u = h[:, :M], h[:, M:]
    return (_act(g) * u) @ down.T


def _per_expert_mean_output(x, gate_up, down):
    """Mean output per expert. x:[T,H], gate_up:[E,2M,H], down:[E,H,M] -> [E,H]."""
    E = gate_up.shape[0]
    outs = torch.empty(E, x.shape[-1], dtype=torch.float32, device=x.device)
    for e in range(E):
        outs[e] = _expert_out(x, gate_up[e].float(), down[e].float()).mean(0)
    return outs


def _load_calib_ids(tok, corpus, max_tokens, device, pack_one=False):
    rows = [json.loads(ln) for ln in open(corpus) if ln.strip()]
    txts = [(r.get("text") or r.get("prompt") or "") for r in rows if (r.get("text") or r.get("prompt"))]
    if pack_one:
        # one long sequence: concat raw chat-formatted texts until max_tokens
        buf = []
        n = 0
        for t in txts:
            ids = tok(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            if n + ids.shape[0] > max_tokens and n > 0:
                break
            buf.append(ids)
            n += ids.shape[0]
        seq = torch.cat(buf)[:max_tokens].unsqueeze(0).to(device)
        return [seq]
    out = []
    for t in txts:
        ids = tok(t, add_special_tokens=False, return_tensors="pt",
                  truncation=True, max_length=max_tokens)["input_ids"].to(device)
        out.append(ids)
    return out


def _fwd(model, ids, device):
    with torch.no_grad():
        model(input_ids=ids, attention_mask=torch.ones_like(ids),
              mm_token_type_ids=torch.zeros_like(ids), use_cache=False)


def phase_a_freq(args, device):
    """Teacher routing frequency per (layer, teacher-expert-id)."""
    log(f"[A] load teacher {args.teacher}")
    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager",
        device_map={"": device}).eval()
    E = int(model.config.text_config.num_experts)
    Lmods = sorted([(int(n.split("layers.")[1].split(".")[0]), m)
                    for n, m in model.named_modules() if n.endswith(".experts")],
                   key=lambda t: t[0])
    L = len(Lmods)
    freq = torch.zeros(L, E, dtype=torch.float64)
    cur = {}

    def pre(li):
        def h(_m, a):
            cur[li] = a[1].detach().reshape(-1).cpu()  # top_k_index
        return h

    handles = [m.register_forward_pre_hook(pre(li)) for li, m in Lmods]
    ids_list = _load_calib_ids(tok, args.freq_corpus, args.freq_max_tokens, device)[: args.freq_max_seqs]
    log(f"[A] {len(ids_list)} seqs for freq")
    for i, ids in enumerate(ids_list):
        cur.clear()
        _fwd(model, ids, device)
        for li in range(L):
            freq[li] += torch.bincount(cur[li], minlength=E).double()
        if (i + 1) % 16 == 0:
            log(f"[A]   {i + 1}/{len(ids_list)}")
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    log(f"[A] freq done: total routed events/layer ~{int(freq[0].sum())}")
    return freq.float()


def _assign_dropped_soft(mean_out, keep_ids, drop_ids, topk=2):
    """Soft top-k assignment: each dropped -> top-k survivors, frac = softmax(cosine).
    Returns {survivor_id: [(dropped_id, frac), ...]}."""
    mo = torch.nn.functional.normalize(mean_out.float(), dim=-1)
    surv = torch.tensor(keep_ids)
    assign = {i: [] for i in keep_ids}
    k = min(topk, len(keep_ids))
    for j in drop_ids:
        sims = mo[surv] @ mo[j]
        vals, idx = torch.topk(sims, k)
        fr = torch.softmax(vals, dim=0)
        for t in range(k):
            assign[keep_ids[int(idx[t])]].append((j, float(fr[t])))
    return assign


def fold_layer(x, gu_t, dn_t, keep_ids, drop_ids, freq_l, device, stats, freq_exp=1.0, norm_anchor="survivor", assign_topk=0):
    """Survivor-anchored DERN Eq.11 fold for one layer.
    x:[T,H] (live student input), gu_t/dn_t: teacher experts [128,*], freq_l:[128].
    Returns merged gate_up [98,2M,H], down [98,H,M] in slot order = sorted(keep_ids)."""
    x = x.float()
    mean_out = _per_expert_mean_output(x, gu_t, dn_t)              # [128,H]
    if assign_topk and assign_topk > 1:
        assign = _assign_dropped_soft(mean_out, keep_ids, drop_ids, topk=assign_topk)   # {i:[(j,frac)]}
    else:
        _hard = _assign_dropped(mean_out, keep_ids, drop_ids)                           # {i:[j]}
        assign = {i: [(j, 1.0) for j in _hard[i]] for i in keep_ids}
    new_gu, new_dn = [], []
    scales = []
    for i in keep_ids:                                           # slot order
        _pairs = [(i, 1.0)] + assign[i]
        members = [p[0] for p in _pairs]
        _fr = torch.tensor([p[1] for p in _pairs], device=freq_l.device, dtype=freq_l.dtype)
        w = (freq_l[members].clamp(min=1e-6) ** freq_exp) * _fr
        w = (w / w.sum()).to(device)                            # [m]
        gum = gu_t[members].float()                             # [m,2M,H]
        dnm = dn_t[members].float()                             # [m,H,M]
        mg = (w[:, None, None] * gum).sum(0)                    # [2M,H]  freq-weighted avg
        md = (w[:, None, None] * dnm).sum(0)                    # [H,M]
        # --- DERN Eq.11 norm-equalization (anchor selectable) ---
        if norm_anchor == "members_mean":
            mns = torch.stack([_expert_out(x, gu_t[j].float(), dn_t[j].float()).norm(dim=-1).mean()
                               for j in members])                      # [m] per-member original norm
            ns = (w.to(mns.device) * mns).sum()                        # freq-weighted mean magnitude
        else:                                                          # survivor (baseline)
            ns = _expert_out(x, gu_t[i].float(), dn_t[i].float()).norm(dim=-1).mean()
        o_merg = _expert_out(x, mg, md)                             # merged [T,H]
        nm = o_merg.norm(dim=-1).mean().clamp(min=1e-6)
        s = (ns / nm).item()
        md = md * s                                             # down_proj is the output linear
        scales.append(s)
        new_gu.append(mg.to(torch.bfloat16))
        new_dn.append(md.to(torch.bfloat16))
    stats.append((sum(len(v) for v in assign.values()),
                  float(min(scales)), float(sum(scales) / len(scales)), float(max(scales))))
    return torch.stack(new_gu), torch.stack(new_dn)


def phase_b_merge(args, freq, device):
    log(f"[B] load student {args.student}")
    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager",
        device_map={"": device}).eval()
    km = json.load(open(args.keep_meta))
    keep = {int(k): list(map(int, v)) for k, v in km["keep"].items()}
    drop = {int(k): list(map(int, v)) for k, v in km["drop"].items()}
    tindex = _st_index(args.teacher)
    Lmods = {int(n.split("layers.")[1].split(".")[0]): m
             for n, m in model.named_modules() if n.endswith(".experts")}
    stats = []

    def pre(li):
        def h(mod, a):
            x = a[0].detach()                                    # [T,H] live, merged-upstream
            gu_t = _load_tensor(args.teacher, EXP % (li, "gate_up_proj"), tindex, device)
            dn_t = _load_tensor(args.teacher, EXP % (li, "down_proj"), tindex, device)
            mg, md = fold_layer(x, gu_t, dn_t, keep[li], drop[li], freq[li], device, stats,
                                freq_exp=args.freq_exponent, norm_anchor=args.norm_anchor,
                                assign_topk=args.assign_topk)
            with torch.no_grad():
                mod.gate_up_proj.copy_(mg)                       # overwrite IN PLACE
                mod.down_proj.copy_(md)
            del gu_t, dn_t, mg, md
            torch.cuda.empty_cache()
        return h

    handles = [Lmods[li].register_forward_pre_hook(pre(li)) for li in sorted(Lmods)]
    seq = _load_calib_ids(tok, args.seq_corpus, args.seq_max_tokens, device, pack_one=True)[0]
    log(f"[B] sequential merge forward: {seq.shape[1]} tokens, {len(Lmods)} layers")
    _fwd(model, seq, device)
    for h in handles:
        h.remove()
    log("[B] per-layer fold (n_dropped, scale min/mean/max):")
    for li, (nd, smn, smu, smx) in enumerate(stats):
        if li % 3 == 0 or smn < 0.8 or smx > 1.25:
            log(f"    L{li:2d}: folded={nd:2d}  s=[{smn:.3f} {smu:.3f} {smx:.3f}]")

    # finite check
    bad = []
    for li, m in Lmods.items():
        if not (torch.isfinite(m.gate_up_proj).all() and torch.isfinite(m.down_proj).all()):
            bad.append(li)
    if bad:
        raise SystemExit(f"FATAL non-finite merged experts at layers {bad}")

    # generation canary (AR-first, off-manifold rule)
    log("[B] generation canary ...")
    p = tok.apply_chat_template([{"role": "user", "content": "Write a Python function that returns the nth Fibonacci number."}],
                                add_generation_prompt=True, tokenize=False)
    enc = tok(p, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=80, do_sample=False,
                           mm_token_type_ids=torch.zeros_like(enc["input_ids"]))
    txt = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    log(f"[B] canary out[:200]: {txt[:200]!r}")

    log(f"[B] save_pretrained -> {args.out}")
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    # copy aux (tokenizer/template/processor) from student
    for fn in os.listdir(args.student):
        s = os.path.join(args.student, fn)
        if os.path.isfile(s) and not fn.endswith(".safetensors") and "index" not in fn:
            import shutil
            shutil.copy2(s, os.path.join(args.out, fn))
    log("[B] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--keep-meta", required=True)
    ap.add_argument("--freq-corpus", required=True)
    ap.add_argument("--seq-corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freq-max-seqs", type=int, default=64)
    ap.add_argument("--freq-max-tokens", type=int, default=4096)
    ap.add_argument("--seq-max-tokens", type=int, default=8192)
    ap.add_argument("--freq-exponent", type=float, default=1.0)
    ap.add_argument("--norm-anchor", default="survivor", choices=["survivor", "members_mean"])
    ap.add_argument("--assign-topk", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device
    freq = phase_a_freq(args, device)
    phase_b_merge(args, freq, device)


if __name__ == "__main__":
    main()
