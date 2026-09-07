#!/usr/bin/env python
"""EOG emit-map for Qwen3.6-35B-A3B (T202 machinery, ported to this tokenizer).

WHAT IT MEASURES
  For every MoE layer, which experts carry the routing mass at positions that PREDICT an
  end-of-generation token (<|im_end|> 248046 / <|endoftext|> 248044) -- i.e. positions t
  where input_ids[t+1] is EOG. That is the "terminator emit" position: the hidden state at
  t is what the LM head turns into an EOG logit, and the experts routed at t are what built
  it.

WHY A BACKGROUND ARM IS MANDATORY
  T202 ranked by raw emit-position RMS. That is confounded: a globally-hot expert tops the
  emit list simply by being hot everywhere. So we accumulate the SAME statistic over all
  NON-emit positions and report

      lift[L][e] = (emit weight share of e) / (background weight share of e)

  A lift > 1 means the expert is preferentially recruited when the model is about to stop.
  Raw emit mass is kept too, so the confounded ranking can be reproduced and compared.

BASIS
  Corpus is the SAME router calibration corpus that produced competence_qwen35b_coder_lcbmpe
  -- the map our drop-map ranking is computed from. Emit-map and saliency are therefore
  commensurable; a different corpus would not be.

ROUTER CAPTURE
  Forward hooks on the per-layer MoE gate Linear (out_features == num_experts). No reliance
  on output_router_logits, which is not uniformly plumbed for this arch.

GPU DISCIPLINE
  CUDA_VISIBLE_DEVICES must already be exported by the caller. This script REFUSES to run if
  it is unset -- polling free memory is not reserving a GPU (bs2 GPU0 is not ours).
"""
import argparse
import json
import os
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/ml/models/Qwen3.6-35B-A3B")
    ap.add_argument("--corpus", default="/srv/ml/repos/omnimergekit/recipes/"
                                        "qwen3_6_35b_a3b_prune/results/"
                                        "router_calib_corpus_coder_lcbmpe_qwen.jsonl")
    ap.add_argument("--eog-ids", default="248046,248044")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="0 = all docs")
    ap.add_argument("--out", default="/mnt/sdc/ream-work/eog_emit_map_qwen36.json")
    args = ap.parse_args()

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit("REFUSING: CUDA_VISIBLE_DEVICES is not exported. bs2 GPU0 is not ours; "
                 "export CUDA_VISIBLE_DEVICES=1 before running.")
    print("CUDA_VISIBLE_DEVICES=%s" % os.environ["CUDA_VISIBLE_DEVICES"])

    from transformers import AutoModelForCausalLM, AutoTokenizer

    eog = set(int(x) for x in args.eog_ids.split(","))
    tok = AutoTokenizer.from_pretrained(args.model)
    print("loading %s ..." % args.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    model.eval()
    print("loaded in %.0fs" % (time.time() - t0))

    # ---- locate the MoE gates -----------------------------------------------------------
    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    E = int(tcfg.num_experts)
    K = int(tcfg.num_experts_per_tok)
    # Qwen3_5MoeTopKRouter.forward -> (softmax probs, renormalised top-k weights, indices).
    # Hook it and consume the model's OWN weights/indices; never recompute the routing.
    gates = [(n, m) for n, m in model.named_modules()
             if type(m).__name__ == "Qwen3_5MoeTopKRouter"]

    def layer_idx(name):
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
        raise SystemExit("REFUSING: cannot parse a layer index out of gate name %r" % name)

    gates.sort(key=lambda nm: layer_idx(nm[0]))
    idx_seen = [layer_idx(n) for n, _ in gates]
    if idx_seen != sorted(set(idx_seen)):
        sys.exit("REFUSING: duplicate/!monotonic gate layer indices %s" % idx_seen[:12])
    print("found %d MoE gates (E=%d, K=%d) layers %d..%d"
          % (len(gates), E, K, idx_seen[0], idx_seen[-1]) if gates else "found 0 MoE gates")
    if not gates:
        sys.exit("REFUSING: no MoE gate modules matched -- do not guess, inspect the model.")
    for n, _ in gates[:2]:
        print("   e.g. %s" % n)

    nL = len(gates)
    dev = "cuda:0"
    emit_w = torch.zeros(nL, E, dtype=torch.float64, device=dev)
    bg_w = torch.zeros(nL, E, dtype=torch.float64, device=dev)
    emit_c = torch.zeros(nL, E, dtype=torch.float64, device=dev)
    bg_c = torch.zeros(nL, E, dtype=torch.float64, device=dev)

    cap = {}           # layer_idx -> router logits for the current doc
    mask_holder = {}   # "emit" -> bool tensor over positions

    def mk_hook(li):
        def hook(_m, _inp, out):
            # Qwen3_5MoeTopKRouter -> (probs, router_scores, router_indices). Use the
            # model's OWN scores/indices; recomputing them would be a second implementation
            # that could silently disagree with what the experts were actually weighted by.
            _, scores, idx = out
            w = scores.reshape(-1, K).float()
            idx = idx.reshape(-1, K).long()
            m = mask_holder["emit"]
            if w.shape[0] != m.shape[0]:
                raise RuntimeError("gate %d saw %d rows, mask has %d"
                                   % (li, w.shape[0], m.shape[0]))
            src_w = torch.zeros(w.shape[0], E, device=w.device).scatter_(1, idx, w)
            src_c = torch.zeros(w.shape[0], E, device=w.device).scatter_(
                1, idx, torch.ones_like(w))
            emit_w[li] += src_w[m].sum(0).double()
            bg_w[li] += src_w[~m].sum(0).double()
            emit_c[li] += src_c[m].sum(0).double()
            bg_c[li] += src_c[~m].sum(0).double()
        return hook

    handles = [mod.register_forward_hook(mk_hook(i)) for i, (_, mod) in enumerate(gates)]

    rows = [json.loads(x) for x in open(args.corpus) if x.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print("corpus rows=%d" % len(rows))

    n_emit_pos = n_pos = 0
    t0 = time.time()
    with torch.inference_mode():
        for i, r in enumerate(rows):
            ids = tok(r["text"], add_special_tokens=False, return_tensors="pt")["input_ids"]
            if ids.shape[1] > args.max_tokens:          # keep the TAIL: the final EOG lives there
                ids = ids[:, -args.max_tokens:]
            if ids.shape[1] < 2:
                continue
            ids = ids.to(dev)
            nxt = ids[0, 1:]
            m = torch.zeros(ids.shape[1], dtype=torch.bool, device=dev)
            m[:-1] = torch.isin(nxt, torch.tensor(sorted(eog), device=dev))
            if not m.any():
                continue
            mask_holder["emit"] = m
            model(input_ids=ids, use_cache=False)
            n_emit_pos += int(m.sum())
            n_pos += int(m.numel())
            if (i + 1) % 50 == 0:
                print("  [%4d/%d] emit_pos=%d  total_pos=%d  %.0fs"
                      % (i + 1, len(rows), n_emit_pos, n_pos, time.time() - t0), flush=True)
    for h in handles:
        h.remove()

    # ---- sanity: emit accumulation must be a SMALL fraction of total --------------------
    tot_emit = float(emit_c.sum())
    tot_bg = float(bg_c.sum())
    exp_emit = n_emit_pos * K * nL
    print("\n[sanity] emit positions=%d  background positions=%d" % (n_emit_pos, n_pos - n_emit_pos))
    print("[sanity] emit routings=%.0f (expect %d = emit_pos*K*L)  bg routings=%.0f"
          % (tot_emit, exp_emit, tot_bg))
    if abs(tot_emit - exp_emit) > max(1.0, 0.001 * exp_emit):
        sys.exit("REFUSING: emit accumulation != emit_pos*K*L -- the emit mask did not gate "
                 "the accumulation. Do not trust this map.")
    print("[sanity] EMIT_MASK_OK")

    ew, bw = emit_w.cpu(), bg_w.cpu()
    es = ew / ew.sum(1, keepdim=True).clamp_min(1e-12)
    bs = bw / bw.sum(1, keepdim=True).clamp_min(1e-12)
    lift = (es / bs.clamp_min(1e-12))

    out = {
        "model": args.model, "corpus": args.corpus, "eog_ids": sorted(eog),
        "n_layers": nL, "n_experts": E, "top_k": K,
        "n_emit_positions": n_emit_pos, "n_positions": n_pos,
        "gate_names": [n for n, _ in gates],
        "emit_weight": ew.tolist(), "bg_weight": bw.tolist(),
        "emit_count": emit_c.cpu().tolist(), "bg_count": bg_c.cpu().tolist(),
        "lift": lift.tolist(),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print("wrote %s" % args.out)
    print(">>> EOG_EMIT_MAP_OK layers=%d experts=%d emit_pos=%d" % (nL, E, n_emit_pos))


if __name__ == "__main__":
    main()
