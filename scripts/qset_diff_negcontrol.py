"""Negative control for qset_map_diff.py.

The identity control (map A vs itself -> rho=1.0, 0 diff) proves the comparator returns
"identical" for identical input. It does NOT prove the comparator can DETECT a difference
-- a tool hardcoded to report "identical" passes it too. This compares two DIFFERENT
languages inside the SAME map: if rust and php cannot be separated, then an A-vs-B verdict
of "identical" means nothing.
"""
import importlib.util
import json
import sys

MAP = sys.argv[1] if len(sys.argv) > 1 else "/mnt/sdc/agentbench/tierb/map_qsetA.json"
TOOL = "/mnt/sdc/harness/qset_map_diff.py"

# import the SHIPPED helpers so the control exercises the real code path
src = open(TOOL).read().split("ap = argparse.ArgumentParser()")[0]
ns = {}
exec(compile(src, TOOL, "exec"), ns)
spearman, score_of = ns["spearman"], ns["score_of"]

d = json.load(open(MAP))["categories"]
print("=== NEGATIVE CONTROL: two DIFFERENT languages inside the SAME map ===")
print("    (if the tool cannot separate rust from php, an A-vs-B 'identical' is worthless)")
print("  %-34s %8s %10s %11s" % ("pair", "rho", "topK ovl", "drop diff"))
pairs = [("targeted_swe_rust", "targeted_swe_php"),
         ("targeted_swe_java", "targeted_swe_javascript"),
         ("targeted_swe_go", "targeted_swe_ruby")]
worst = 0.0
for a, b in pairs:
    rhos, ovls, diffs = [], [], []
    for li in sorted(d[a], key=int):
        ea = {e["id"]: e for e in d[a][li]}
        eb = {e["id"]: e for e in d[b][li]}
        ids = sorted(set(ea) & set(eb))
        va = [score_of(ea[i], "tc") for i in ids]
        vb = [score_of(eb[i], "tc") for i in ids]
        rhos.append(spearman(va, vb))
        ta = set(sorted(ids, key=lambda i: -score_of(ea[i], "tc"))[:30])
        tb = set(sorted(ids, key=lambda i: -score_of(eb[i], "tc"))[:30])
        ovls.append(len(ta & tb) / 30)
        da = set(sorted(ids, key=lambda i: (score_of(ea[i], "tc"), i))[:30])
        db = set(sorted(ids, key=lambda i: (score_of(eb[i], "tc"), i))[:30])
        diffs.append(len(da ^ db) // 2)
    r = sum(rhos) / len(rhos)
    worst = max(worst, r)
    print("  %-34s %8.4f %9.1f%% %11.2f"
          % (a.replace("targeted_swe_", "") + " vs " + b.replace("targeted_swe_", ""),
             r, 100 * sum(ovls) / len(ovls), sum(diffs) / len(diffs)))
print()
if worst < 0.999:
    print("  CONTROL PASSES — the comparator separates genuinely different categories,")
    print("  so an A-vs-B verdict of 'identical' would be a real finding.")
else:
    print("  CONTROL FAILS — the comparator cannot tell two different languages apart.")
    print("  Do NOT trust any A-vs-B verdict from it until this is fixed.")
