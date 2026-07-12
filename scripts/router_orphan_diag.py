#!/usr/bin/env python3
"""T180 router-orphan diagnostic.

Hypothesis (the "abstain path is overridden by renorm" mechanism): the tokens
that loop in the pruned A2 (62e) model are exactly the tokens whose softmax mass,
in the UNPRUNED 128e router, lands on DROPPED experts. Pruning removes those
experts from the softmax denominator, so the surviving experts -- which the 128e
router scored near-zero -- get renormalized to full unit weight (line 1311) and
inject a confident WRONG MoE contribution that the always-on dense MLP can't
overcome.

This measures, on the 128e model (which has ALL experts), per token:
  - surviving_mass = sum of softmax_128(router_probabilities)[A2_keep] : how much
    of the router's mass the A2 prune RETAINS. Low => token is "orphaned" (its
    experts were dropped) => 62e renorm inflates it by 1/surviving_mass.
  - top8_survivor_overlap = |top8(128e) ∩ A2_keep| / 8 : did the 128e router's
    actual choices survive the prune?

Compared across two prompt buckets from loop_screen_sample.jsonl:
  - multilingual (the loopers in A2)  vs  openended (clean; 128e loops 0/56).

If multilingual tokens show systematically LOWER surviving_mass / overlap than
openended tokens, the mechanism is confirmed and median surviving_mass on the
clean bucket gives the abstain threshold tau for a confidence-gated fix.

Prefill only (no generation): the multilingual content is in the prompt; orphaned
routing manifests on those input tokens. Captures router_probabilities via a
forward hook on every Gemma4TextRouter (robust, no output_router_logits plumbing).
Run on bs2 GPU0, omk python.
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(m):
    print("[orphan %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/ml/models/base/gemma-4-26B-A4B-it")
    ap.add_argument("--keep-meta",
                    default="/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/google/gemma-4-A4B-62e-fc15_25-p8-it/expert_drop_metadata.json",
                    help="A2 expert_drop_metadata.json with per_layer_keep")
    ap.add_argument("--sample", default="/mnt/sdc/ml/corpora/loop_screen_sample.jsonl")
    ap.add_argument("--per-bucket", type=int, default=25)
    ap.add_argument("--out", default="/srv/ml/eval_results_tracks_2_3/t176_phase3/router_orphan_diag.json")
    args = ap.parse_args()

    keep_meta = json.load(open(args.keep_meta))
    per_layer_keep = {int(k): sorted(v) for k, v in keep_meta["per_layer_keep"].items()}
    L = len(per_layer_keep)
    E_keep = len(per_layer_keep[0])
    log("A2 keep-map: %d layers, %d experts kept/layer" % (L, E_keep))

    rows = [json.loads(x) for x in open(args.sample)]
    ml = [r for r in rows if r.get("bucket") == "multilingual"][:args.per_bucket]
    oe = [r for r in rows if r.get("bucket") == "openended"][:args.per_bucket]
    log("prompts: multilingual=%d  openended=%d" % (len(ml), len(oe)))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": 0}).eval()

    # locate the per-layer routers and hook them to capture router_probabilities
    # (index 0 of the router's return = softmax over all 128 experts, pre-topk).
    routers = []
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "Gemma4TextRouter":
            routers.append((name, mod))
    routers.sort(key=lambda nm: int(nm[0].split("layers.")[1].split(".")[0]))
    log("found %d routers" % len(routers))
    assert len(routers) == L, "router count %d != keep-map layers %d" % (len(routers), L)

    cap = {}

    def mk_hook(li):
        def hook(_m, _inp, out):
            # out = (router_probabilities, top_k_weights, top_k_index)
            cap[li] = out[0].detach().float().cpu()  # [tokens, 128]
        return hook

    handles = [routers[li][1].register_forward_hook(mk_hook(li)) for li in range(L)]

    def run_prompt(p):
        cap.clear()
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)
        enc = tok([text], return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            model(**enc)
        # per layer: surviving_mass per token, top8 overlap per token
        surv_layer = []   # mean over tokens, per layer
        ov_layer = []
        surv_all_tokens = []  # flattened token-level (layer-avg) for distribution
        ntok = None
        for li in range(L):
            probs = cap[li][0] if cap[li].dim() == 3 else cap[li]  # [tokens,128]
            ntok = probs.shape[0]
            keep = per_layer_keep[li]
            surv = probs[:, keep].sum(dim=-1)               # [tokens]
            top8 = torch.topk(probs, k=8, dim=-1).indices    # [tokens,8]
            keepset = torch.zeros(probs.shape[1], dtype=torch.bool)
            keepset[keep] = True
            ov = keepset[top8].float().mean(dim=-1)          # [tokens] frac of top8 surviving
            surv_layer.append(surv.mean().item())
            ov_layer.append(ov.mean().item())
            surv_all_tokens.append(surv.numpy())
        # token-level: average surviving mass across layers for each token
        tok_surv = np.mean(np.stack(surv_all_tokens, 0), axis=0)  # [tokens]
        return dict(surv_layer=surv_layer, ov_layer=ov_layer,
                    tok_surv=tok_surv.tolist(), ntok=ntok)

    def agg(bucket_rows, label):
        all_surv_layer = np.zeros(L)
        all_ov_layer = np.zeros(L)
        tok_surv_pool = []
        for i, r in enumerate(bucket_rows):
            res = run_prompt(r["prompt"])
            all_surv_layer += np.array(res["surv_layer"])
            all_ov_layer += np.array(res["ov_layer"])
            tok_surv_pool.extend(res["tok_surv"])
            if (i + 1) % 5 == 0:
                log("  [%s] %d/%d" % (label, i + 1, len(bucket_rows)))
        n = len(bucket_rows)
        tsp = np.array(tok_surv_pool)
        out = dict(
            n_prompts=n, n_tokens=int(tsp.size),
            surv_mass_mean=float(tsp.mean()), surv_mass_median=float(np.median(tsp)),
            surv_mass_p10=float(np.percentile(tsp, 10)),
            surv_mass_p90=float(np.percentile(tsp, 90)),
            inflation_median=float(1.0 / max(np.median(tsp), 1e-6)),
            surv_layer_mean=(all_surv_layer / n).tolist(),
            ov_layer_mean=(all_ov_layer / n).tolist(),
            frac_tokens_orphaned_below_0_3=float((tsp < 0.3).mean()),
            frac_tokens_orphaned_below_0_5=float((tsp < 0.5).mean()),
        )
        log("[%s] surv_mass median=%.3f mean=%.3f p10=%.3f  inflation(med)=%.1fx  "
            "frac<0.3=%.2f  top8-overlap(mean)=%.3f" % (
                label, out["surv_mass_median"], out["surv_mass_mean"], out["surv_mass_p10"],
                out["inflation_median"], out["frac_tokens_orphaned_below_0_3"],
                float(np.mean(out["ov_layer_mean"]))))
        return out

    t0 = time.time()
    res_ml = agg(ml, "multilingual")
    res_oe = agg(oe, "openended")
    for h in handles:
        h.remove()

    summary = {
        "model": args.model, "keep_meta": args.keep_meta,
        "n_layers": L, "experts_kept": E_keep,
        "multilingual": res_ml, "openended": res_oe,
        "separation": {
            "surv_mass_median_ml": res_ml["surv_mass_median"],
            "surv_mass_median_oe": res_oe["surv_mass_median"],
            "ratio_oe_over_ml": res_oe["surv_mass_median"] / max(res_ml["surv_mass_median"], 1e-6),
        },
        "wall_s": round(time.time() - t0, 0),
    }
    json.dump(summary, open(args.out, "w"), indent=1)
    log("DONE -> %s" % args.out)
    log("SEPARATION: openended median surv_mass=%.3f vs multilingual=%.3f (%.1fx)" % (
        res_oe["surv_mass_median"], res_ml["surv_mass_median"],
        summary["separation"]["ratio_oe_over_ml"]))


if __name__ == "__main__":
    main()
