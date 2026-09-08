#!/usr/bin/env python
"""Re-read the HE+/MPE screen from summary.json for every REAM arm.

Truncation comes from token_stats.finish_reasons.length -- the SERVER's own report -- not
from OMK_CAP_CHECK, which under-counts on this backend (it compares re-tokenized length
against max_gen_toks and never reaches the ceiling).
"""
import json
import os

R = "/srv/ml/eval_results/ream_arms"


def cell(t, n):
    p = f"{R}/{t}/{n}/summary.json"
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    ts = s.get("token_stats") or {}
    fr = ts.get("finish_reasons") or {}
    return s.get("score"), fr.get("length", 0), ts.get("n"), (s.get("sampler") or {}).get("name")


names = sorted(set(os.listdir(f"{R}/humaneval_full_think")) | set(os.listdir(f"{R}/multipl_e_100")))
hdr = ("cell", "MPE-100", "HE+", "HE+ len-term", "sampler")
print("%-42s %9s %9s %16s  %s" % hdr)
print("-" * 96)
rows = []
for n in names:
    m = cell("multipl_e_100", n)
    h = cell("humaneval_full_think", n)
    if not (m or h):
        continue
    rows.append((n, m, h))
rows.sort(key=lambda r: -((r[1] or (0,))[0] or 0))
for n, m, h in rows:
    ms = "%.4f" % m[0] if m and m[0] is not None else "-"
    hs = "%.4f" % h[0] if h and h[0] is not None else "-"
    hc = "%d/%d (%.1f%%)" % (h[1], h[2], 100 * h[1] / h[2]) if h and h[2] else "-"
    sm = (m or h)[3]
    print("%-42s %9s %9s %16s  %s" % (n, ms, hs, hc, sm))
