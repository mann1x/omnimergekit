#!/usr/bin/env python3
"""Print T201 score + completion-token percentiles per cell from summary.json."""
import json
import glob
import os
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/mnt/sdc/ml/sampler_probe_t201"
ORDER = [
    "128e_greedy_bud", "128e_greedy_nobud", "128e_gemma_bud", "128e_gemma_nobud",
    "v7_greedy_bud", "v7_greedy_nobud", "v7_gemma_bud", "v7_gemma_nobud",
    "v7_gemma_bud_minp", "v7_gemma_bud_rep", "v7_gemma_bud_both",
    "v7_gemma_nothink_none", "v7_gemma_nothink_rep",
]

hdr = ("cell", "score", "cp50", "cp90", "cmax", "empty")
print("%-24s %6s %6s %6s %7s %5s" % hdr)
for c in ORDER:
    sj = glob.glob(os.path.join(P, c, "**", "summary.json"), recursive=True)
    if not sj:
        print("%-24s %6s" % (c, "PENDING"))
        continue
    j = json.load(open(sj[0]))
    ts = j.get("token_stats") or {}
    ct = ts.get("completion_tokens") or {}
    sc = j.get("score")
    sc = ("%.3f" % sc) if isinstance(sc, float) else str(sc)
    print("%-24s %6s %6s %6s %7s %5s" % (
        c, sc, str(ct.get("p50")), str(ct.get("p90")),
        str(ct.get("max")), str(ts.get("empty_completions"))))
