#!/usr/bin/env python3
"""One line per EVAL record of trainer logs:  bonsai_eval_table.py RUN.out [RUN.out ...]

Prints full KL (kl*) when present, else the top-K KL (klk*), for every eval set, plus wiki top-1 and trit drift."""
import json
import sys

for f in sys.argv[1:]:
    print("==", f.split("/")[-2] if "/" in f else f)
    for line in open(f):
        if not line.startswith("EVAL "):
            continue
        d = json.loads(line[5:])
        out = [f"{d['tag']:9s}"]
        for pre in ("kl", "klk"):
            ks = [k for k in d if k == pre or (k.startswith(pre + "_") and len(k) == len(pre) + 2)]
            if ks:
                out.append(pre + " " + " ".join(f"{d[k]:.3f}" for k in sorted(ks)))
        out.append(f"top1 {d.get('top1', 0):.3f}")
        if "mask_change" in d:
            out.append(f"mask {100 * d['mask_change']:.2f}% flip {100 * d['sign_flip']:.4f}%")
        print("  " + " | ".join(out))
