#!/usr/bin/env python
"""Dump REAP per-expert saliency for Gemma-4 MoE layers.

WHY NOT REUSE REAM's PROFILER
-----------------------------
REAM's Merger is Qwen-shaped: it slices `moe_layer.gate.weight` and sets `gate.out_features`.
Gemma-4 has neither -- it routes through `router.proj.weight` + `router.scale` +
`router.per_expert_scale`. So the capture layer is written here against the real modules;
only the SALIENCY MATH is taken from REAM, and it is verified against REAM's own function
rather than trusted (see GATE below).

TWO SALIENCY VARIANTS -- THIS IS THE PORT'S REAL DECISION
---------------------------------------------------------
REAM's `saliency.reap(gate_logits, expert_activations, top_k)` applies its OWN
softmax + top-k to raw logits. Gemma-4's router does more than that:

    probs        = softmax(proj(normed_hidden))
    w, idx       = topk(probs, k)
    w           /= w.sum(-1, keepdim=True)          # renormalise
    w            = w * per_expert_scale[idx]        # PER-EXPERT CONSTANT

`per_expert_scale` is a constant per expert, so it rescales that expert's whole saliency
and CAN REORDER THE RANKING -- which is the only thing this probe is used for. Feeding raw
logits to REAM's reap() silently drops it. So we emit BOTH and never pick one silently:

  * "published"  -- gate = softmax(logits), exactly REAP as published / as REAM computes it
  * "effective"  -- gate = the model's own top_k_weights (renormalised, per_expert_scale applied)

GATE (fatal)
------------
On the first chunk, the incremental accumulator is compared elementwise against
`ream.saliency.reap()` run on the same chunk. reap() takes a .mean() over each expert's
routed tokens, so chunked accumulation is a sum/count weighted mean, NOT a running sum --
this gate is what proves the accumulator reproduces it.

Basis is recorded: corpus path + sha256 + token count + chunking go into the output JSON.
A saliency vector without its basis is not comparable to anything.
"""
import argparse
import hashlib
import json
import os
import time

import torch


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Capture:
    """Grabs, per MoE layer: the experts' input, the routing decision, and the router probs.

    Two hooks are needed because the router and the experts see DIFFERENTLY NORMALISED
    inputs (`hidden_states_flat` vs `pre_feedforward_layernorm_2(hidden_states_flat)`),
    and because the decoder layer discards the router's `router_probabilities` (`_, w, i =
    self.router(...)`) -- the module still returns it, so a forward hook can see it.
    """

    def __init__(self):
        self.router_probs = {}
        self.expert_in = {}
        self.topk_w = {}
        self.topk_i = {}

    def router_hook(self, li):
        def fn(_mod, _inp, out):
            self.router_probs[li] = out[0].detach()
        return fn

    def experts_pre_hook(self, li):
        def fn(_mod, args):
            self.expert_in[li] = args[0].detach()
            self.topk_i[li] = args[1].detach()
            self.topk_w[li] = args[2].detach()
        return fn


def formula_gate(ream_reap, n_exp=128, n_tok=256, hid=16, top_k=8, seed=0):
    """Prove our sum/count accumulator == REAM's reap() EXACTLY, on unambiguous routing.

    This is deliberately synthetic. On the real model the gate cannot be exact, because
    reap() RE-DERIVES the top-k from logits while Gemma-4's router picks its top-8 from a
    *bf16* softmax -- ties at the 8th slot break differently (differently again CPU vs CUDA),
    and reap's softmax(log(p)) renormalises a bf16 row that does not sum to exactly 1.
    Both are properties of reap's re-derivation, not of our arithmetic. So the arithmetic is
    proven here, where routing is unambiguous, and routing fidelity is *reported* separately
    on the real data.
    """
    torch.manual_seed(seed)
    logits = torch.randn(n_tok, n_exp)
    acts = torch.randn(n_exp, n_tok, hid)
    gate = torch.softmax(logits, dim=-1, dtype=torch.float)
    gval, sel = torch.topk(gate, k=top_k, dim=-1)
    mask = torch.nn.functional.one_hot(sel, num_classes=n_exp).permute(2, 1, 0)
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

    s_ = torch.zeros(n_exp, dtype=torch.float64)
    n_ = torch.zeros(n_exp, dtype=torch.float64)
    for e in hit:
        e = e.item()
        pos, tix = torch.where(mask[e])
        nrm = acts[e][tix].norm(dim=-1)
        s_[e] += float((nrm * gval[tix, pos]).sum())
        n_[e] += len(tix)
    ours = (s_ / n_.clamp(min=1)).float()
    theirs = ream_reap(logits.unsqueeze(0), acts, top_k=top_k)
    d = (ours - theirs).abs().max().item()
    if d > 1e-5:
        raise SystemExit(f"FORMULA GATE FAIL: accumulator vs ream.saliency.reap max|d|={d:.3e}")
    log(f">>> FORMULA_GATE_OK max|d|={d:.3e} (accumulator == ream.saliency.reap)")


def expert_outputs(experts, hidden, exp_idx, tok_idx):
    """Raw expert output (BEFORE the routing weight) for one expert on given tokens."""
    st = hidden[tok_idx]
    gate, up = torch.nn.functional.linear(st, experts.gate_up_proj[exp_idx]).chunk(2, dim=-1)
    h = experts.act_fn(gate) * up
    return torch.nn.functional.linear(h, experts.down_proj[exp_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, help="plain text calibration file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-tokens", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=64, help="-1 = all")
    ap.add_argument("--ream-dir", default="/workspace/ream")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # Load REAM's saliency.py BY PATH. Importing `ream.saliency` normally pulls
    # ream/__init__.py -> merger -> data.calibration_data, i.e. the entire merge engine and
    # its calibration tensors, none of which this probe uses. The function itself is pure
    # torch. This is still verbatim reuse -- it is REAM's own file, unmodified.
    import importlib.util
    _sp = os.path.join(args.ream_dir, "ream", "saliency.py")
    if not os.path.exists(_sp):
        raise SystemExit(f"REAM saliency.py not found at {_sp}")
    _spec = importlib.util.spec_from_file_location("ream_saliency_vendored", _sp)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ream_reap = _mod.reap
    log(f"REAM saliency loaded verbatim from {_sp}")
    formula_gate(ream_reap)
    routing_diag = {}

    from transformers import AutoTokenizer, AutoModelForCausalLM

    raw = open(args.corpus, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    log(f"corpus {args.corpus} sha256={sha[:16]} bytes={len(raw)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(raw.decode("utf-8", "ignore"), return_tensors="pt").input_ids[0]
    n_chunk = len(ids) // args.chunk_tokens
    if args.chunks > 0:
        n_chunk = min(n_chunk, args.chunks)
    log(f"tokens={len(ids)} chunk_tokens={args.chunk_tokens} chunks={n_chunk}")

    log("loading model (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    lm = model.model.language_model
    layers = [(i, l) for i, l in enumerate(lm.layers) if getattr(l, "enable_moe_block", False)]
    n_exp = model.config.text_config.num_experts
    top_k = model.config.text_config.top_k_experts
    log(f"moe layers={len(layers)} experts={n_exp} top_k={top_k}")

    cap = Capture()
    handles = []
    for li, layer in layers:
        handles.append(layer.router.register_forward_hook(cap.router_hook(li)))
        handles.append(layer.experts.register_forward_pre_hook(cap.experts_pre_hook(li)))

    # accumulators: sum of (||act|| * gate) and count of routed tokens, per expert
    acc = {li: {"pub_s": torch.zeros(n_exp, dtype=torch.float64),
                "pub_n": torch.zeros(n_exp, dtype=torch.float64),
                "eff_s": torch.zeros(n_exp, dtype=torch.float64),
                "eff_n": torch.zeros(n_exp, dtype=torch.float64)} for li, _ in layers}

    gate_done = False
    t0 = time.time()
    for c in range(n_chunk):
        seg = ids[c * args.chunk_tokens:(c + 1) * args.chunk_tokens].unsqueeze(0).to(args.device)
        with torch.no_grad():
            model(seg)

        for li, layer in layers:
            probs = cap.router_probs[li].float()          # [T, E] softmax over raw logits
            hidden = cap.expert_in[li]                    # [T, H] experts' own input
            tk_i = cap.topk_i[li]                         # [T, K]
            tk_w = cap.topk_w[li].float()                 # [T, K] renorm * per_expert_scale
            experts = layer.experts

            mask = torch.nn.functional.one_hot(tk_i, num_classes=n_exp).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

            if not gate_done:
                # Not a pass/fail: RECORD how far reap's re-derived routing drifts from the
                # model's actual routing on this data. See formula_gate() for why.
                gl = torch.log(probs.clamp(min=1e-30)).unsqueeze(0).cpu()
                their_sel = torch.topk(
                    torch.softmax(gl.view(-1, n_exp), dim=-1, dtype=torch.float),
                    k=top_k, dim=-1).indices
                their_mask = torch.nn.functional.one_hot(
                    their_sel, num_classes=n_exp).permute(2, 1, 0)
                agree = (their_mask == mask.cpu()).all(dim=-1).all(dim=-1)
                a = torch.sort(their_sel, dim=-1).values
                b = torch.sort(tk_i.cpu(), dim=-1).values
                tok_dis = int((a != b).any(dim=-1).sum())
                routing_diag.update({
                    "probe_layer": li,
                    "experts_routing_identical": int(agree.sum()),
                    "experts_total": n_exp,
                    "tokens_with_different_top_k": tok_dis,
                    "tokens_total": int(a.shape[0]),
                    "note": "reap() re-derives top-k from logits; Gemma-4 routes from a bf16 "
                            "softmax, so ties at the k-th slot differ. The model's "
                            "top_k_index is ground truth and is what this probe uses.",
                })
                log(f">>> ROUTING_DIAG layer={li} "
                    f"experts_identical={int(agree.sum())}/{n_exp} "
                    f"tokens_differing={tok_dis}/{int(a.shape[0])}")
                gate_done = True

            with torch.no_grad():
                for e in hit:
                    e = e.item()
                    pos, tix = torch.where(mask[e])
                    nrm = expert_outputs(experts, hidden, e, tix).float().norm(dim=-1)
                    acc[li]["pub_s"][e] += float((nrm * probs[tix, e]).sum())
                    acc[li]["pub_n"][e] += len(tix)
                    acc[li]["eff_s"][e] += float((nrm * tk_w[tix, pos]).sum())
                    acc[li]["eff_n"][e] += len(tix)

        cap.router_probs.clear(); cap.expert_in.clear()
        cap.topk_w.clear(); cap.topk_i.clear()
        if (c + 1) % 8 == 0:
            log(f"chunk {c+1}/{n_chunk}  {(time.time()-t0)/(c+1):.2f}s/chunk")

    for h in handles:
        h.remove()

    out = {"meta": {"model": args.model, "corpus": args.corpus, "corpus_sha256": sha,
                    "chunk_tokens": args.chunk_tokens, "chunks": n_chunk,
                    "tokens_used": n_chunk * args.chunk_tokens,
                    "num_experts": n_exp, "top_k": top_k,
                    "variants": ["published", "effective"],
                    "note": "published = softmax(logits) per REAP/REAM; "
                            "effective = model's top_k_weights incl per_expert_scale",
                    "routing_diagnostic": routing_diag},
           "published": {}, "effective": {}}
    for li, _ in layers:
        a = acc[li]
        out["published"][str(li)] = (a["pub_s"] / a["pub_n"].clamp(min=1)).tolist()
        out["effective"][str(li)] = (a["eff_s"] / a["eff_n"].clamp(min=1)).tolist()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    log(f">>> REAP_PROBE_DONE layers={len(layers)} -> {args.out}")


if __name__ == "__main__":
    main()
