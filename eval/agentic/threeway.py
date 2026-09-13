"""Three-way contrast on ONE frozen task set: no cap / cap silent / cap announced.

Cells differ in the reasoning budget only:
  nocap      --reasoning-budget -1
  cap        --reasoning-budget 8192
  cap+msg    --reasoning-budget 8192 --reasoning-budget-message "<think_budget_message>"
Everything else (GGUF, libllama, template, -c/-np, max_tokens, greedy sampler) is
identical across all three, and every cell is restricted to the SAME task ids, so
composition cannot move any column.

Scores come from the sample score, never recomputed. Loops come from the canonical
detector in loop_detect.py -- never redeclared here.
"""
import sys, json, math, statistics as st
sys.path.insert(0, "/srv/ml/repos/omnimergekit/eval/agentic")
from inspect_ai.log import read_eval_log
from loop_detect import detect, message_texts

MSG_NEEDLE = "I have used my thinking budget"

def load(path, only):
    log = read_eval_log(path)
    out = {}
    for s in (log.samples or []):
        sid = str(s.id)
        if sid not in only:
            continue
        score = None
        for v in (s.scores or {}).values():
            val = getattr(v, "value", None)
            if isinstance(val, (int, float)): score = float(val)
            elif val in ("C", "I"):           score = 1.0 if val == "C" else 0.0
            if score is not None: break
        calls = []
        for ev in (s.events or []):
            if getattr(ev, "event", None) != "model": continue
            u = getattr(getattr(ev, "output", None), "usage", None)
            n = getattr(u, "output_tokens", None) if u else None
            if n is not None: calls.append(int(n))
        looped = False; worst = 0; nmsg = 0; blocks = 0
        for m in (s.messages or []):
            if getattr(m, "role", None) != "assistant": continue
            for t in message_texts(m):
                blocks += 1
                if MSG_NEEDLE in t: nmsg += 1
                d = detect(t)
                if d.get("looped"): looped = True
                worst = max(worst, int(d.get("sentence_reps") or 0))
        out[sid] = dict(score=score, calls=calls, out=sum(calls), looped=looped,
                        worst=worst, nmsg=nmsg, blocks=blocks,
                        secs=getattr(s, "working_time", None))
    return out, log.status

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2)

def med(v): return st.median(v) if v else 0

ids = set(json.load(open(sys.argv[1])))
cells = []
for spec in sys.argv[2:]:
    label, path = spec.split("=", 1)
    d, status = load(path, ids)
    cells.append((label, d, status))

common = set.intersection(*[set(d) for _, d, _ in cells])
common = {k for k in common if all(d[k]["score"] is not None for _, d, _ in cells)}
print("=== THREE-WAY, frozen id set: %d requested, %d scored in ALL cells ===" % (len(ids), len(common)))
for label, d, status in cells:
    print("  %-10s status=%-9s loaded=%d" % (label, status, len(d)))
print()
print("%-10s %6s %9s %9s %9s %9s %7s %7s %7s" %
      ("cell", "pass", "out p50", "out sum", "calls", ">=8192", ">=32k", "loops", "worst"))
for label, d, _ in cells:
    v = [d[k] for k in sorted(common)]
    allc = [c for x in v for c in x["calls"]]
    print("%-10s %3d/%-3d %9d %9d %9d %9d %7d %6d/%-3d %6d" %
          (label, sum(1 for x in v if x["score"] >= 1.0), len(v),
           med([x["out"] for x in v]), sum(x["out"] for x in v), len(allc),
           sum(1 for c in allc if c >= 8192), sum(1 for c in allc if c >= 32768),
           sum(1 for x in v if x["looped"]), len(v), max([x["worst"] for x in v] or [0])))
print()
inj = [(l, sum(d[k]["nmsg"] for k in common)) for l, d, _ in cells]
print("  cap-message blocks per cell:", ", ".join("%s=%d" % t for t in inj))
print()
base = cells[0]
for label, d, _ in cells[1:]:
    b = sum(1 for k in common if base[1][k]["score"] >= 1.0 and d[k]["score"] < 1.0)
    c = sum(1 for k in common if base[1][k]["score"] < 1.0 and d[k]["score"] >= 1.0)
    r = [d[k]["out"] / base[1][k]["out"] for k in common if base[1][k]["out"] > 0]
    print("  %s vs %s: pass discordant %d/%d McNemar p=%.4f%s | median output ratio %.3f"
          % (label, base[0], b, c, mcnemar(b, c),
             "  UNDERPOWERED" if b + c < 10 else "", med(r)))
