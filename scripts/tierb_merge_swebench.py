"""Merge the b1/b2/b2r Tier-B trace files into one pool, refusing on overlap.

b2r re-ran 10 instances that batch 2 had already dispatched. All 10 failed in b2 (empty
patch), so their PASS traces should be disjoint from b2's -- but "should be" is not a
check. A duplicated instance id would enter the pool twice and be weighted twice, which
is a silent corruption of the map, so this ABORTS on any overlap rather than deduping
quietly and letting the arm-of-origin question disappear.

Also reports the per-class census against the Tier-B targets, because the pool is
consumed per class (the classes ARE the log_parser families) and a pooled total hides a
class that is short.
"""
import json
import sys
from collections import Counter, defaultdict

SRC = [
    ("b1", "/mnt/sdc/agentbench/tierb/tierb_b1_128k.json"),
    ("b2", "/mnt/sdc/agentbench/tierb/tierb_b2_128k.json"),
    ("b2r", "/mnt/sdc/agentbench/tierb/tierb_b2r_128k.json"),
]
OUT = "/mnt/sdc/agentbench/tierb/tierb_swebench_pool.json"
TARGETS = (5, 8)

REQUIRED = {"task_id", "bench", "budget_capped", "messages", "resolved", "set", "weight"}

loaded, envelope = {}, None
for tag, path in SRC:
    with open(path) as fh:
        d = json.load(fh)
    missing = REQUIRED - set(d["traces"][0]) if d["traces"] else REQUIRED
    if missing:
        sys.exit("ABORT: %s traces lack expected field(s): %s — schema changed, "
                 "fix this script rather than letting the census read nothing" % (tag, sorted(missing)))
    loaded[tag] = d
    if envelope is None:
        envelope = {k: v for k, v in d.items() if k != "traces"}
    print("  %-4s %3d traces   keys=%s" % (tag, len(d["traces"]), sorted(d)))

# --- overlap gate ----------------------------------------------------------
seen, origin, dupes = set(), {}, []
for tag, _ in SRC:
    for t in loaded[tag]["traces"]:
        iid = t["task_id"]
        if iid in seen:
            dupes.append((iid, origin[iid], tag))
        seen.add(iid)
        origin.setdefault(iid, tag)
print()
if dupes:
    print("ABORT — %d instance id(s) appear in more than one arm:" % len(dupes))
    for iid, a, b in dupes:
        print("   %-40s in %s and %s" % (iid, a, b))
    print("Pooling them would double-weight the instance. Decide which arm owns each,")
    print("then re-run with the loser removed.")
    sys.exit(1)
print("overlap gate: PASS — %d unique instances, no id in two arms" % len(seen))

# --- merge -----------------------------------------------------------------
merged = list(envelope.items())
traces = []
for tag, _ in SRC:
    for t in loaded[tag]["traces"]:
        t = dict(t)
        t["arm"] = tag           # provenance: which run produced this trace
        traces.append(t)
out = dict(merged)
out["traces"] = traces
with open(OUT, "w") as fh:
    json.dump(out, fh)
print("wrote %s  (%d traces)" % (OUT, len(traces)))

# --- per-class census ------------------------------------------------------
print()
print("=== per-class census (classes ARE the log_parser families) ===")
byc = Counter(t["bench"] for t in traces)
byarm = defaultdict(Counter)
for t in traces:
    byarm[t["bench"]][t["arm"]] += 1
print("  %-16s %6s %8s %8s %8s   %s" % ("class", "have", "need5", "need8", "arms", ""))
short5 = short8 = 0
for c in sorted(byc):
    n = byc[c]
    d5, d8 = max(0, TARGETS[0] - n), max(0, TARGETS[1] - n)
    short5 += d5 > 0
    short8 += d8 > 0
    print("  %-16s %6d %8s %8s   %s" % (
        c, n, "OK" if d5 == 0 else "-%d" % d5, "OK" if d8 == 0 else "-%d" % d8,
        dict(byarm[c])))
print("  %-16s %6d" % ("TOTAL", sum(byc.values())))
print()
print("  classes short of %d: %d      classes short of %d: %d"
      % (TARGETS[0], short5, TARGETS[1], short8))

# --- budget-cut composition ------------------------------------------------
cut = sum(1 for t in traces if t["budget_capped"])
print()
print("=== thinking-budget cuts in the pool ===")
print("  traces containing a cut: %d/%d" % (cut, len(traces)))
print("  (do NOT --exclude-capped: on b2 that is ~half the corpus, and the cut traces")
print("   are PASS traces -- they solved the instance despite being interrupted.)")
