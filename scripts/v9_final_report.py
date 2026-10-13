"""Final v9 vs b24_msg2 anchor report. Reads summary.json .score (never raw results),
then the full paired table + per-problem fix/break list."""
import json, os, sys
from math import comb

V9SUM = "/mnt/sdc/lcb_v6_v9/v9_b24_msg2/lcb_v6_77q/v9_b24_msg2/summary.json"
V9SAMP = "/mnt/sdc/lcb_v6_v9/v9_b24_msg2/lcb_v6_77q/v9_b24_msg2/lcb_result.samples.jsonl"
ASUM = "/mnt/sdc/lcb_v6_budget_probe/b24_msg2/lcb_v6_77q/b24_msg2/summary.json"
ASAMP = "/mnt/sdc/lcb_v6_budget_probe/b24_msg2/lcb_v6_77q/b24_msg2/lcb_result.samples.jsonl"

if not os.path.exists(V9SUM):
    sys.exit("summary.json not written yet")

v9s, ans = json.load(open(V9SUM)), json.load(open(ASUM))
print("=== SCORES (from summary.json .score) ===")
print("  v9-agentic  : %.4f   metric=%s sampler=%s"
      % (v9s["score"], v9s.get("metric"), (v9s.get("sampler") or {}).get("name")))
print("  b24_msg2    : %.4f   metric=%s sampler=%s"
      % (ans["score"], ans.get("metric"), (ans.get("sampler") or {}).get("name")))
print("  delta       : %+.4f (%+.2f pp)" % (v9s["score"] - ans["score"],
                                            100 * (v9s["score"] - ans["score"])))

def load(p):
    d = {}
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        d[r["task_id"]] = bool(r.get("passed"))
    return d

v9, an = load(V9SAMP), load(ASAMP)
sh = sorted(set(v9) & set(an))
fixes = [k for k in sh if not an[k] and v9[k]]
breaks = [k for k in sh if an[k] and not v9[k]]
both = sum(1 for k in sh if an[k] and v9[k])
neither = sum(1 for k in sh if not an[k] and not v9[k])
n01, n10 = len(fixes), len(breaks)
n = n01 + n10
p = 1.0 if n == 0 else min(1.0, sum(comb(n, i) for i in range(0, min(n01, n10) + 1)) / (2**n) * 2)

print("\n=== PAIRED (n=%d shared) ===" % len(sh))
print("  both pass %d | both fail %d | v9 fixes %d | v9 breaks %d" % (both, neither, n01, n10))
print("  McNemar exact p = %.3f%s" % (p, "  *SIGNIFICANT*" if p < 0.05 else "  (not significant)"))
print("\n  v9 FIXES  : %s" % [k.split('/')[-1] for k in fixes])
print("  v9 BREAKS : %s" % [k.split('/')[-1] for k in breaks])
print("  both FAIL : %s" % [k.split('/')[-1] for k in sh if not an[k] and not v9[k]])

fin = v9s.get("finish") or v9s.get("finish_reasons")
print("\n=== v9 generation health ===")
print("  finish: %s" % fin)
ts = v9s.get("token_stats") or {}
if ts:
    print("  token_stats: %s" % {k: ts[k] for k in list(ts)[:6]})
