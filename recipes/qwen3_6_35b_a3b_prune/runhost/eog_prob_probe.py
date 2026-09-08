#!/usr/bin/env python
"""DIRECT measurement of premature-EOG propensity. No routing proxy.

The routing instrument (eog_emit_map + eog_crosscheck + eog_arm_correlation) came back
negative, but it is a weak instrument for a pruned arm: it scores BASE expert ids on BASE
routing, while a pruned model re-normalises its top-8 over a different expert set. So stop
proxying and measure the thing itself.

MultiPL-E runs raw-completion: the prompt ends mid-function and the model must continue
writing code. A "silent empty" is the model terminating instead. So the quantity that IS
the failure is

    P(EOG | MPE prompt)      at the very first generated position

measured per arm on the SAME 300 prompts. Two readings:
  p0      -- P(EOG) at the first generated position
  p1      -- P(EOG) after forcing a single newline (the observed empty completion is '\\n',
             so this is the position at which the arm actually stopped)

and the split that matters: p0 restricted to the problems that arm ACTUALLY returned empty
vs the ones it answered. If premature EOG is the mechanism, the empty problems carry
visibly higher p0 for that arm -- and armI's whole distribution sits above armD's.

Per-arm, single process (loop in the shell) so nothing shares a CUDA allocator.
"""
import argparse
import glob
import json
import os
import sys

import torch

LANGS = ("humaneval-rs", "humaneval-java", "humaneval-js")
GEN = ("/srv/ml/eval_results/ream_arms/multipl_e_100/%s/generations")
EOG = [248046, 248044]


def load_prompts(cell):
    out = {}
    for ld in LANGS:
        for p in sorted(glob.glob(os.path.join(GEN % cell, ld, "*.json"))):
            d = json.load(open(p))
            out["%s::%s" % (ld.split("-")[1], d["name"])] = d["prompt"]
    return out


def load_empties(cell):
    p = "/srv/ml/eval_results/ream_arms/multipl_e_100/%s/mpe_result.samples.jsonl" % cell
    em = set()
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        c = d.get("completion") or ""
        if not c.strip():
            em.add(d.get("task_id"))
    return em


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--cell", required=True, help="results cell name for prompts + empties")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit("REFUSING: CUDA_VISIBLE_DEVICES not exported (bs2 GPU0 is not ours).")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    prompts = load_prompts(args.cell)
    empties = load_empties(args.cell)
    print("%s: %d prompts, %d empty completions on this arm"
          % (args.label, len(prompts), len(empties)))
    if len(prompts) != 300:
        sys.exit("REFUSING: expected 300 prompts, got %d" % len(prompts))

    tok = AutoTokenizer.from_pretrained("/srv/ml/models/Qwen3.6-35B-A3B")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True).eval()
    nl = tok("\n", add_special_tokens=False)["input_ids"]

    rec = {}
    with torch.inference_mode():
        for i, (tid, pr) in enumerate(sorted(prompts.items())):
            ids = tok(pr, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
            lg = model(input_ids=ids, use_cache=False).logits[0, -1].float()
            p = torch.softmax(lg, -1)
            p0 = float(sum(p[t] for t in EOG))
            ids2 = torch.cat([ids, torch.tensor([nl], device=ids.device)], dim=1)
            lg2 = model(input_ids=ids2, use_cache=False).logits[0, -1].float()
            p2 = torch.softmax(lg2, -1)
            p1 = float(sum(p2[t] for t in EOG))
            rec[tid] = {"p0": p0, "p1": p1, "empty": tid in empties}
            if (i + 1) % 100 == 0:
                print("  [%d/300]" % (i + 1), flush=True)

    import statistics as st
    allp0 = [v["p0"] for v in rec.values()]
    allp1 = [v["p1"] for v in rec.values()]
    e0 = [v["p0"] for v in rec.values() if v["empty"]]
    n0 = [v["p0"] for v in rec.values() if not v["empty"]]
    print("\n%s  p0(EOG at first gen pos): mean=%.5f p50=%.5f p90=%.5f max=%.4f"
          % (args.label, st.mean(allp0), st.median(allp0),
             sorted(allp0)[int(.9 * len(allp0))], max(allp0)))
    print("%s  p1(after forced newline): mean=%.5f p50=%.5f max=%.4f"
          % (args.label, st.mean(allp1), st.median(allp1), max(allp1)))
    if e0:
        print("%s  p0 on ACTUALLY-EMPTY (n=%d): mean=%.5f p50=%.5f"
              % (args.label, len(e0), st.mean(e0), st.median(e0)))
        print("%s  p0 on answered      (n=%d): mean=%.5f p50=%.5f"
              % (args.label, len(n0), st.mean(n0), st.median(n0)))
    json.dump({"label": args.label, "model": args.model, "rec": rec}, open(args.out, "w"))
    print(">>> EOG_PROB_OK %s -> %s" % (args.label, args.out))


if __name__ == "__main__":
    main()
