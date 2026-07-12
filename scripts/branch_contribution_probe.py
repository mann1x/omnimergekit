#!/usr/bin/env python3
"""T181 branch-contribution probe — MoE branch vs dense-MLP "base" branch.

Gemma-4 each layer (modeling_gemma4.py:1376-1388) sums an ALWAYS-ON dense MLP
branch and the 128->62 MoE branch, EACH independently RMS-normed:
    h1 = post_feedforward_layernorm_1(mlp(x))        # dense "base" branch
    h2 = post_feedforward_layernorm_2(experts(x))    # MoE branch
    out = h1 + h2
Because each branch is RMS-normed before summing, post-norm MAGNITUDE is ~token-
independent (the MoE can't "dominate by magnitude" -> why PES alpha-sweeps failed).
The corruption from incapable surviving experts must therefore be DIRECTIONAL.

We measure, per layer, on the COMPLETION region of A2's own greedy generation,
for looping (multilingual) vs clean (openended) prompts, on A2 (and 128e base as
reference, teacher-forced on the same sequence):
  - raw pre-norm RMS of mlp() and experts() outputs (saturation check; normed away
    downstream but diagnostic of expert blow-up).
  - COS(h1, h2): cosine between the post-norm dense and MoE branches. Hypothesis:
    on loop tokens the MoE branch turns ORTHOGONAL/ANTI-aligned to the dense "base"
    branch (steering the residual off-manifold), vs ALIGNED on clean tokens. A
    per-token COS(h1,h2) that craters on loops is the OUTPUT-CONDITIONED abstain
    signal a router-gate metric can't see (gate entropy/mass were ~equal, T176.4/T180).

Run on bs2 GPU0 (both models fit: A2 ~28GB + base ~49GB < 97GB). omk python.
"""
import argparse
import json
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
    print("[branch %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def load(model_dir, dev):
    m = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": dev})
    m.eval()
    cap = {}

    def mk(tag, li):
        def hook(_mod, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[(tag, li)] = o.detach()
        return hook

    handles = []
    for name, mod in m.named_modules():
        cls = mod.__class__.__name__
        if name.endswith(".mlp") and cls == "Gemma4TextMLP":
            li = int(name.split("layers.")[1].split(".")[0])
            handles.append(mod.register_forward_hook(mk("dense_raw", li)))
        elif name.endswith(".experts") and cls == "Gemma4TextExperts":
            li = int(name.split("layers.")[1].split(".")[0])
            handles.append(mod.register_forward_hook(mk("moe_raw", li)))
        elif name.endswith(".post_feedforward_layernorm_1"):
            li = int(name.split("layers.")[1].split(".")[0])
            handles.append(mod.register_forward_hook(mk("h1", li)))
        elif name.endswith(".post_feedforward_layernorm_2"):
            li = int(name.split("layers.")[1].split(".")[0])
            handles.append(mod.register_forward_hook(mk("h2", li)))
    return m, cap, handles


@torch.no_grad()
def branch_stats(model, cap, ids, dev, start, nlayers):
    cap.clear()
    model(input_ids=ids, attention_mask=torch.ones_like(ids),
          mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
    cos_layers, ratio_layers, moe_rms_layers, dense_rms_layers = [], [], [], []
    for li in range(nlayers):
        h1 = cap.get(("h1", li))
        h2 = cap.get(("h2", li))
        draw = cap.get(("dense_raw", li))
        mraw = cap.get(("moe_raw", li))
        if h1 is None or h2 is None:
            continue
        h1 = h1.reshape(-1, h1.shape[-1]).float()[start:]
        h2 = h2.reshape(-1, h2.shape[-1]).float()[start:]
        cos = F.cosine_similarity(h1, h2, dim=-1)            # [T] post-norm direction
        cos_layers.append(cos)
        if draw is not None and mraw is not None:
            dr = draw.reshape(-1, draw.shape[-1]).float()[start:]
            mr = mraw.reshape(-1, mraw.shape[-1]).float()[start:]
            d_rms = dr.pow(2).mean(-1).sqrt()
            m_rms = mr.pow(2).mean(-1).sqrt()
            dense_rms_layers.append(d_rms)
            moe_rms_layers.append(m_rms)
            ratio_layers.append(m_rms / d_rms.clamp_min(1e-6))
    cos = torch.stack(cos_layers).mean(0)                    # per token, mean over layers
    out = {"cos_h1h2": cos.mean().item(), "cos_per_layer": torch.stack(cos_layers).mean(1).cpu().tolist()}
    if ratio_layers:
        ratio = torch.stack(ratio_layers).mean(0)
        out["moe_dense_rms_ratio"] = ratio.mean().item()
        out["moe_rms"] = torch.stack(moe_rms_layers).mean().item()
        out["dense_rms"] = torch.stack(dense_rms_layers).mean().item()
        out["ratio_per_layer"] = torch.stack(ratio_layers).mean(1).cpu().tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2", default=A2)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--no-base", dest="with_base", action="store_false",
                    help="skip 128e base (A2-only; needed when A2+base exceed one GPU)")
    ap.add_argument("--sample", default="/mnt/sdc/ml/corpora/loop_screen_sample.jsonl")
    ap.add_argument("--per-bucket", type=int, default=12)
    ap.add_argument("--gen-tokens", type=int, default=1024)
    ap.add_argument("--out", default="/srv/ml/eval_results_tracks_2_3/t176_phase3/branch_contribution.json")
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.sample)]
    ml = [r["prompt"] for r in rows if r.get("bucket") == "multilingual"][:args.per_bucket]
    oe = [r["prompt"] for r in rows if r.get("bucket") == "openended"][:args.per_bucket]
    prompts = [("multilingual", p) for p in ml] + [("openended", p) for p in oe]
    log("prompts: multilingual=%d openended=%d" % (len(ml), len(oe)))

    tok = AutoTokenizer.from_pretrained(args.a2, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    log("loading A2 -> cuda:0")
    a2m, a2cap, a2h = load(args.a2, 0)
    nlayers = a2m.config.text_config.num_hidden_layers
    bm = bcap = None
    if args.with_base:
        log("loading base 128e -> cuda:0")
        bm, bcap, bh = load(args.base, 0)

    results = []
    for bucket, prompt in prompts:
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        penc = tok(chat, return_tensors="pt", add_special_tokens=False)
        plen = penc["input_ids"].shape[1]
        with torch.no_grad():
            gen = a2m.generate(**{k: v.to("cuda:0") for k, v in penc.items()},
                               max_new_tokens=args.gen_tokens, do_sample=False,
                               repetition_penalty=1.0, use_cache=True,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        comp = gen[0][plen:]
        looped = detect_loop(tok.decode(comp, skip_special_tokens=True))
        full = gen[0:1].to("cuda:0")
        a2s = branch_stats(a2m, a2cap, full, 0, plen, nlayers)
        rec = {"bucket": bucket, "looped": bool(looped), "comp_len": int(comp.numel()), "a2": a2s}
        if bm is not None:
            rec["base"] = branch_stats(bm, bcap, full, 0, plen, nlayers)
        results.append(rec)
        bb = rec.get("base", {})
        log("%-12s loop=%-5s len=%4d | A2 cos=%.3f ratio=%.2f | base cos=%.3f ratio=%.2f" % (
            bucket, looped, comp.numel(), a2s["cos_h1h2"], a2s.get("moe_dense_rms_ratio", 0),
            bb.get("cos_h1h2", 0), bb.get("moe_dense_rms_ratio", 0)))

    for h in a2h + (bh if bm is not None else []):
        h.remove()

    import statistics as st

    def grp(pred, key, sub):
        v = [r[key][sub] for r in results if pred(r) and sub in r.get(key, {})]
        return round(st.mean(v), 4) if v else None

    a2_loop = lambda r: r["looped"] and r["bucket"] == "multilingual"  # noqa: E731
    a2_clean = lambda r: (not r["looped"]) and r["bucket"] == "openended"  # noqa: E731
    agg = {
        "n_total": len(results),
        "n_loop_ml": sum(1 for r in results if a2_loop(r)),
        "n_clean_oe": sum(1 for r in results if a2_clean(r)),
        "A2_cos_loop": grp(a2_loop, "a2", "cos_h1h2"),
        "A2_cos_clean": grp(a2_clean, "a2", "cos_h1h2"),
        "A2_ratio_loop": grp(a2_loop, "a2", "moe_dense_rms_ratio"),
        "A2_ratio_clean": grp(a2_clean, "a2", "moe_dense_rms_ratio"),
        "base_cos_loop": grp(a2_loop, "base", "cos_h1h2"),
        "base_cos_clean": grp(a2_clean, "base", "cos_h1h2"),
    }
    json.dump({"agg": agg, "per_prompt": results}, open(args.out, "w"), indent=1)
    log("=" * 60)
    for k, v in agg.items():
        log("  %-18s %s" % (k, v))
    if agg["A2_cos_loop"] is not None and agg["A2_cos_clean"] is not None:
        log("SIGNAL: A2 cos(h1,h2) loop=%.3f vs clean=%.3f (delta %+.3f)" % (
            agg["A2_cos_loop"], agg["A2_cos_clean"], agg["A2_cos_loop"] - agg["A2_cos_clean"]))
    log("-> %s" % args.out)


if __name__ == "__main__":
    main()
