"""Per-MODEL-CALL completion-token distribution for a cell.

The question is how often a generation runs into a ceiling, so the unit must be
the individual model call, not the per-task sum. Two ceilings exist:
  reasoning budget (8192 in the capped cells, -1 = none in the unbounded cells)
  max_tokens      (32768 in every cell)
Restricting to a shared id list keeps composition out of the comparison.
"""
import sys, json
from inspect_ai.log import read_eval_log

def calls(path, only=None):
    log = read_eval_log(path)
    out = []
    ids = []
    for s in (log.samples or []):
        sid = str(s.id)
        if only is not None and sid not in only:
            continue
        ids.append(sid)
        for ev in (s.events or []):
            if getattr(ev, "event", None) != "model":
                continue
            u = getattr(getattr(ev, "output", None), "usage", None)
            if u is None:
                continue
            n = getattr(u, "output_tokens", None)
            if n is not None:
                out.append(int(n))
    return out, ids

def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]

only = None
if len(sys.argv) > 1 and sys.argv[1] != "-":
    only = set(json.load(open(sys.argv[1])))
cells = sys.argv[2:]
print("%-26s %7s %7s %7s %7s %8s  %14s %14s" %
      ("cell", "calls", "p50", "p90", "p99", "max", ">=8192", ">=32768"))
for spec in cells:
    label, path = spec.split("=", 1)
    v, ids = calls(path, only)
    if not v:
        print("%-26s  (no calls)" % label); continue
    n = len(v)
    g8 = sum(1 for x in v if x >= 8192)
    g32 = sum(1 for x in v if x >= 32768)
    print("%-26s %7d %7d %7d %7d %8d  %6d (%5.2f%%) %6d (%5.2f%%)"
          % (label, n, q(v, .5), q(v, .9), q(v, .99), max(v), g8, 100*g8/n, g32, 100*g32/n))
    print("%-26s   tasks=%d" % ("", len(ids)))
