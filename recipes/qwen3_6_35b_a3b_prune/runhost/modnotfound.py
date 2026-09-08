import json, glob, os, collections
mods = collections.Counter()
percell = collections.Counter()
for f in sorted(glob.glob("/srv/ml/eval_results/*/lcb_v6_77q*/*/lcb_result.samples.jsonl")):
    cell = os.path.basename(os.path.dirname(f))
    for line in open(f, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("passed"):
            continue
        r = d.get("reason") or ""
        if "ModuleNotFoundError" in r or "ImportError" in r:
            mods[r.strip()[:80]] += 1
            percell[cell] += 1
print("banked LCB failures caused by a MISSING PYTHON MODULE (not the model):")
for r, n in mods.most_common():
    print("  %3d  %s" % (n, r))
print()
print("per cell:")
for c, n in sorted(percell.items()):
    print("  %-32s %d" % (c, n))
print("\ntotal:", sum(mods.values()), "across", len(percell), "cells")
