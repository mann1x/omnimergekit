"""Does a long generation RUMINATE-TO-FAIL (unbounded) or RUMINATE-AND-STILL-PASS (capped)?

Tests the claim directly, within one arm, on the same task list, with only the
reasoning budget changed:
  1) paired accuracy, capped vs unbounded (McNemar exact)
  2) crosstab: did the task contain a call at/near the ceiling, and did it pass?
A ceiling call is >=8192 output tokens in the capped cell (budget + the answer it
is forced to write afterwards) and >=32768 in the unbounded cell (max_tokens).
"""
import sys, math
from inspect_ai.log import read_eval_log

def load(path):
    log = read_eval_log(path)
    out = {}
    for s in (log.samples or []):
        sid = str(s.id)
        sc = None
        if s.scores:
            for v in s.scores.values():
                val = getattr(v, "value", None)
                if isinstance(val, (int, float)):
                    sc = float(val)
                elif val in ("C", "I"):
                    sc = 1.0 if val == "C" else 0.0
                if sc is not None:
                    break
        calls = []
        for ev in (s.events or []):
            if getattr(ev, "event", None) != "model":
                continue
            u = getattr(getattr(ev, "output", None), "usage", None)
            if u is not None and getattr(u, "output_tokens", None) is not None:
                calls.append(int(u.output_tokens))
        out[sid] = (sc, calls)
    return out

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)

cap = load(sys.argv[1]); unb = load(sys.argv[2]); arm = sys.argv[3]
CAP_CEIL, UNB_CEIL = 8192, 32768
shared = sorted(set(cap) & set(unb))
shared = [k for k in shared if cap[k][0] is not None and unb[k][0] is not None]
print("=== %s : capped-8192 vs unbounded, %d shared scored tasks ===" % (arm, len(shared)))

pc = sum(1 for k in shared if cap[k][0] >= 1.0)
pu = sum(1 for k in shared if unb[k][0] >= 1.0)
b = sum(1 for k in shared if cap[k][0] >= 1.0 and unb[k][0] < 1.0)
c = sum(1 for k in shared if cap[k][0] < 1.0 and unb[k][0] >= 1.0)
print("  pass: capped %d/%d (%.1f%%)   unbounded %d/%d (%.1f%%)"
      % (pc, len(shared), 100*pc/len(shared), pu, len(shared), 100*pu/len(shared)))
print("  discordant capped-only=%d unbounded-only=%d  McNemar exact p=%.4f%s"
      % (b, c, mcnemar(b, c), "   UNDERPOWERED (<10)" if b + c < 10 else ""))

for label, D, ceil in (("capped", cap, CAP_CEIL), ("unbounded", unb, UNB_CEIL)):
    hit = [k for k in shared if any(x >= ceil for x in D[k][1])]
    nohit = [k for k in shared if k not in hit]
    ph = sum(1 for k in hit if D[k][0] >= 1.0)
    pn = sum(1 for k in nohit if D[k][0] >= 1.0)
    print("  %-9s ceiling>=%5d : %3d tasks hit it, pass %d/%d (%.1f%%) | "
          "no-hit pass %d/%d (%.1f%%)"
          % (label, ceil, len(hit), ph, len(hit), 100*ph/len(hit) if hit else 0,
             pn, len(nohit), 100*pn/len(nohit) if nohit else 0))
