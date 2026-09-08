#!/usr/bin/env python
"""What token ACTUALLY ends the empty completions? EOG, or a closing brace?

armD's p0/p1 readings expose a competing explanation the EOG framing missed. MultiPL-E
prompts end INSIDE an open function body and the bench's stop_tokens are ["\\n}", "<file_sep>"].
So a completion recorded as '\\n' has (at least) two very different causes:

  (a) PREMATURE EOG   -- model emits '\\n' then the end-of-generation token. A capability/
                         termination defect in the weights. This is what the audit assumed.
  (b) EMPTY BODY      -- model emits '\\n' then '}', closing the function immediately. The
                         harness stop-sequence "\\n}" fires and truncates, leaving '\\n'.
                         NOT a terminator defect at all -- the model wrote a real (wrong,
                         empty) program and the bench faithfully recorded it.

These are indistinguishable in the saved completion and lead to opposite fixes: (a) argues
for a --force-keep on terminator experts, (b) argues the hybrid degraded code-continuation
and the "silent empty" label is simply wrong.

So decode greedily for a few tokens and LOOK. Greedy matches the canonical eval sampler, so
this reproduces what the bench actually did rather than a sample from a different sampler.

Run per arm on that arm's OWN empty problems plus a matched set of problems it answered.
"""
import argparse
import glob
import json
import os
import sys

import torch

LANGS = ("humaneval-rs", "humaneval-java", "humaneval-js")
GEN = "/srv/ml/eval_results/ream_arms/multipl_e_100/%s/generations"
EOG = {248046, 248044}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--n-control", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit("REFUSING: CUDA_VISIBLE_DEVICES not exported (bs2 GPU0 is not ours).")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = {}
    for ld in LANGS:
        for p in sorted(glob.glob(os.path.join(GEN % args.cell, ld, "*.json"))):
            d = json.load(open(p))
            prompts["%s::%s" % (ld.split("-")[1], d["name"])] = d["prompt"]

    empties, answered = [], []
    sp = "/srv/ml/eval_results/ream_arms/multipl_e_100/%s/mpe_result.samples.jsonl" % args.cell
    for line in open(sp):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        (empties if not (d.get("completion") or "").strip() else answered).append(d["task_id"])
    print("%s: %d empty, %d answered" % (args.label, len(empties), len(answered)))

    tok = AutoTokenizer.from_pretrained("/srv/ml/models/Qwen3.6-35B-A3B")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True).eval()

    def rollout(tid):
        ids = tok(prompts[tid], add_special_tokens=False,
                  return_tensors="pt")["input_ids"].cuda()
        got = []
        with torch.inference_mode():
            for _ in range(args.steps):
                lg = model(input_ids=ids, use_cache=False).logits[0, -1]
                nxt = int(lg.argmax())
                got.append(nxt)
                if nxt in EOG:
                    break
                ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
        return got

    def classify(toks):
        if toks and toks[-1] in EOG:
            return "EOG"
        s = tok.decode(toks)
        st = s.strip()
        if st.startswith("}") or st[:2] in ("}\n", "};"):
            return "CLOSE_BRACE"
        return "CONTINUES_CODE"

    res = {}
    counts = {}
    print("\n--- %s : greedy %d-token rollout on its OWN empty problems ---"
          % (args.label, args.steps))
    for tid in sorted(empties):
        t = rollout(tid)
        c = classify(t)
        counts[c] = counts.get(c, 0) + 1
        res[tid] = {"empty": True, "cls": c, "text": tok.decode(t), "ids": t}
        print("  %-42s %-14s %r" % (tid, c, tok.decode(t)[:60]))
    print("  => %s" % counts)

    cc = {}
    for tid in sorted(answered)[:args.n_control]:
        t = rollout(tid)
        c = classify(t)
        cc[c] = cc.get(c, 0) + 1
        res[tid] = {"empty": False, "cls": c, "text": tok.decode(t), "ids": t}
    print("  control (answered, n=%d): %s" % (min(args.n_control, len(answered)), cc))

    json.dump({"label": args.label, "empty_counts": counts, "control_counts": cc,
               "rec": res}, open(args.out, "w"))
    print(">>> ROLLOUT_OK %s" % args.label)


if __name__ == "__main__":
    main()
