#!/usr/bin/env python3
"""T191 E-RankProbe — is a DIFFUSE capability rank-absorbable by 62 survivor experts?

The decisive cheap test before committing the expensive full E-ExpertKD pod run.
Single layer at a time: build A2's router (FROZEN) + A2's surviving experts (TRAINABLE)
using the exact Gemma-4 modules, then KD-fit the survivor MoE block output to the 128e
TEACHER's MoE block output on captured activations, and measure HELD-OUT block-output
divergence BEFORE vs AFTER training.

Interpretation (a CAPACITY/RANK test, not an AR-quality claim — see plan):
  * held-out divergence drops materially  -> the 62-survivor SwiGLU span CAN represent
    the lost function; commit the full E-ExpertKD run.
  * held-out divergence plateaus           -> 62x704 is a hard capacity wall for this
    capability at this budget; declare it out-of-scope and raise the budget.

The gate is held-out divergence (the right metric for a capacity probe), NEVER the
train-set reconstruction loss (off-manifold rule). Held-out = a CONTIGUOUS tail block
of tokens (~ unseen prompts), stricter than interleaving.

Requires a capture WITH the router_in tap:
  redist.py capture --method expert_kd --driver <cap> --corpus <disjoint calib> ...
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts, Gemma4TextRouter

A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"
_P = "model.language_model.layers.%d.%s"


def log(m):
    print("[rankprobe %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def _index(d):
    p = os.path.join(d, "model.safetensors.index.json")
    return json.load(open(p))["weight_map"]


def _get(d, name, index, device):
    from safetensors import safe_open
    with safe_open(os.path.join(d, index[name]), framework="pt", device=device) as f:
        return f.get_tensor(name)


def build_layer(student, li, cfg, index, device):
    """A2's router (frozen) + survivor experts (trainable, fp32) for layer li."""
    router = Gemma4TextRouter(cfg).to(device)
    experts = Gemma4TextExperts(cfg).to(device)
    with torch.no_grad():
        router.proj.weight.copy_(_get(student, _P % (li, "router.proj.weight"), index, device).float())
        router.scale.copy_(_get(student, _P % (li, "router.scale"), index, device).float())
        router.per_expert_scale.copy_(_get(student, _P % (li, "router.per_expert_scale"), index, device).float())
        experts.gate_up_proj.copy_(_get(student, _P % (li, "experts.gate_up_proj"), index, device).float())
        experts.down_proj.copy_(_get(student, _P % (li, "experts.down_proj"), index, device).float())
    router.float().requires_grad_(False)
    experts.float().requires_grad_(True)
    return router, experts


def divergence(student_out, teacher_out):
    num = ((student_out - teacher_out) ** 2).sum(-1)
    den = (teacher_out ** 2).sum(-1).clamp(min=1e-8)
    relmse = float((num / den).mean())
    cos = float(torch.nn.functional.cosine_similarity(student_out, teacher_out, dim=-1).mean())
    return relmse, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=A2)
    ap.add_argument("--capture", required=True, help="capture .pt WITH router_in tap (method=expert_kd)")
    ap.add_argument("--layers", default="5,12,18,25")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--heldout", type=float, default=0.2)
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help=">0 logs held-out relMSE every N steps (convergence curve: "
                         "is the drop training-limited or capacity-limited?)")
    ap.add_argument("--pass-thresh", type=float, default=30.0,
                    help="mean held-out relMSE drop %% above which rank is 'absorbable'")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/srv/ml/redist_work/rankprobe.json")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.student).text_config
    assert cfg.num_experts == 62, cfg.num_experts
    index = _index(args.student)
    cap = torch.load(args.capture, map_location="cpu", weights_only=False)
    data = cap["data"]
    if "router_in" not in data:
        raise SystemExit("FAIL: capture lacks router_in tap — recapture with --method expert_kd")
    layers = [int(x) for x in args.layers.split(",")]
    log("layers=%s steps=%d lr=%g heldout=%.2f tokens/layer=%d"
        % (layers, args.steps, args.lr, args.heldout, data["block_out"][layers[0]].shape[0]))

    results = {}
    for li in layers:
        router, experts = build_layer(args.student, li, cfg, index, args.device)
        rin = data["router_in"][li].float().to(args.device)
        sin = data["swiglu_in"][li].float().to(args.device)
        tgt = data["block_out"][li].float().to(args.device)
        T = rin.shape[0]
        n_tr = int(T * (1 - args.heldout))
        tr = slice(0, n_tr)
        ho = slice(n_tr, T)              # contiguous tail ~ unseen prompts

        # explicit args (no free-var capture) so the trailing `del` can't shadow them
        def fwd(rtr, exp, rin_sl, sin_sl):
            _, tw, ti = rtr(rin_sl)
            return exp(sin_sl, ti, tw)

        with torch.no_grad():
            rb, cb = divergence(fwd(router, experts, rin[ho], sin[ho]), tgt[ho])
        opt = torch.optim.AdamW(experts.parameters(), lr=args.lr)
        curve = []
        for step in range(args.steps):
            opt.zero_grad()
            loss = ((fwd(router, experts, rin[tr], sin[tr]) - tgt[tr]) ** 2).mean()
            loss.backward()
            opt.step()
            if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
                with torch.no_grad():
                    rc, _ = divergence(fwd(router, experts, rin[ho], sin[ho]), tgt[ho])
                d = round(100 * (rb - rc) / max(rb, 1e-9), 1)
                curve.append({"step": step + 1, "heldout_relmse": round(rc, 4), "drop_pct": d})
                log("  L%-2d  step %4d  held-out relMSE %.4f (drop %.1f%%)" % (li, step + 1, rc, d))
        with torch.no_grad():
            ra, ca = divergence(fwd(router, experts, rin[ho], sin[ho]), tgt[ho])
        drop = round(100 * (rb - ra) / max(rb, 1e-9), 1)
        results[li] = {"before_relmse": round(rb, 4), "after_relmse": round(ra, 4),
                       "before_cos": round(cb, 4), "after_cos": round(ca, 4),
                       "heldout_relmse_drop_pct": drop, "train_tokens": n_tr, "heldout_tokens": T - n_tr}
        log("L%-2d  relMSE %.4f->%.4f (drop %.1f%%)  cos %.4f->%.4f"
            % (li, rb, ra, drop, cb, ca))
        del router, experts, rin, sin, tgt
        torch.cuda.empty_cache()

    mean_drop = sum(r["heldout_relmse_drop_pct"] for r in results.values()) / len(results)
    verdict = "ABSORBABLE" if mean_drop >= args.pass_thresh else "CAPACITY_WALL"
    out = {"capture": args.capture, "layers": layers, "steps": args.steps, "lr": args.lr,
           "mean_heldout_relmse_drop_pct": round(mean_drop, 1), "pass_thresh": args.pass_thresh,
           "verdict": verdict, "per_layer": results}
    json.dump(out, open(args.out, "w"), indent=1)
    log("=" * 64)
    log("MEAN held-out relMSE drop = %.1f%%  (thresh %.1f%%)  ->  %s"
        % (mean_drop, args.pass_thresh, verdict))
    if verdict == "ABSORBABLE":
        log("=> 62 survivors CAN represent the lost function; commit full E-ExpertKD.")
    else:
        log("=> 62x704 is a capacity wall for this capability; out-of-scope at this budget.")
    log("RANKPROBE_DONE -> %s" % args.out)


if __name__ == "__main__":
    main()
