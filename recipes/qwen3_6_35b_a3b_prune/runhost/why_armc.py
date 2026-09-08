#!/usr/bin/env python
"""Why armC collapsed and armB did not -- from the layer-0 grouping telemetry alone.

REAM merges each group as a saliency-weighted average:  w = sal[members]/sum(sal[members]).
So the CENTROID'S OWN WEIGHT SHARE decides what the merge actually does:
  * share ~1.0  -> the merge is a near-no-op, the surviving expert is preserved
  * share ~0.3  -> the surviving expert is genuinely averaged away into its absorbed peers

REAP saliency is extremely peaked (centroid orders of magnitude above the dropped tail),
which is presumably why REAM can afford a 16-way group. Our injected saliency uses a
keep_offset (+2.0) on top of normalised competence scores, which is nearly FLAT by
comparison -- same ordering, completely different dynamic range.

This reads the numbers actually logged by the builds; no GPU, no re-run.
"""
import ast
import re
import statistics as st
import sys

LOG = "/mnt/sdc/ream-work/%s.log"
PAT = re.compile(r"group_ind:\s*(\d+)\s*size:\s*(\d+)\s*group:\s*\[([^\]]*)\]\s*saliency:\s*\[([^\]]*)\]")


def parse(arm):
    rows = []
    for line in open(LOG % arm, errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        size = int(m.group(2))
        sal = [float(x) for x in m.group(4).replace(",", " ").split()]
        rows.append((size, sal))
    return rows


def report(arm):
    rows = parse(arm)
    if not rows:
        print(f"{arm}: no grouping telemetry"); return
    sizes = [s for s, _ in rows]
    merged = [(s, sal) for s, sal in rows if s > 1 and sal]
    print(f"\n=== {arm}   groups={len(rows)}  sum(sizes)={sum(sizes)}  "
          f"merged_groups={len(merged)}  max_size={max(sizes)}")
    hist = {}
    for s in sizes:
        hist[s] = hist.get(s, 0) + 1
    print("    size histogram:", dict(sorted(hist.items())))
    if not merged:
        print("    no multi-member groups"); return
    shares, ratios = [], []
    for s, sal in merged:
        tot = sum(sal)
        if tot <= 0:
            continue
        shares.append(sal[0] / tot)                       # centroid's own weight
        tail = sal[1:]
        if tail and max(tail) > 0:
            ratios.append(sal[0] / (sum(tail) / len(tail)))  # centroid : mean member
    if shares:
        print(f"    centroid weight share : min={min(shares):.3f} "
              f"median={st.median(shares):.3f} max={max(shares):.3f}")
        print(f"    -> median {100*st.median(shares):.1f}% of the merged expert is the "
              f"survivor; {100*(1-st.median(shares)):.1f}% is absorbed experts")
    if ratios:
        print(f"    centroid:mean-member saliency ratio : median={st.median(ratios):.1f}x")


for arm in sys.argv[1:] or ["armB", "armC"]:
    report(arm)

print("\nInterpretation: a high centroid weight share means REAM's merge barely moves the")
print("surviving expert. A low share means the survivor is genuinely averaged with its")
print("absorbed peers -- which is what a 16-way average of unrelated FFNs does to quality.")
