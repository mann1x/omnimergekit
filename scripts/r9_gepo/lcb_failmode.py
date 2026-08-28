"""Why does gepo2 lose on the SAMPLED LCB cohort when it won on greedy?

Hypothesis under test: the brevity training shortens generation. Under greedy at 49152
that HELPED, because this family degenerates into ~180k-char runaway loops and shorter
output meant fewer cap-truncations (gepo2 7/77 vs armJ 15/77). Under the canonical
sampler at 32768 nobody loops (0-5 truncations), so the same shortening now has nothing
harmful to cut and may instead be terminating legitimate reasoning early.

Prediction if true: gepo2's FAILURES should be SHORTER than its passes and shorter than
armJ's output on the same problems, with failures skewed toward no-code / empty /
syntax rather than genuine wrong-answer.
"""
import collections, json, os, re

R = "/srv/ml/eval_results/qwen_suite/lcb_v6_77q"
CELLS = [("armJ", "qwenhybridp24_q6k"), ("gepo1", "qwena3bgepo1_q6k"), ("gepo2", "qwena3bgepo2_q6k")]
CODE = re.compile(r"```|def |class ")


def load(n):
    out = {}
    with open(os.path.join(R, n, "lcb_result.samples.jsonl")) as fh:
        for l in fh:
            if l.strip():
                d = json.loads(l)
                out[d.get("task_id") or d.get("doc_id")] = d
    return out


A = {t: load(n) for t, n in CELLS}


def bucket(r):
    r = (r or "").strip().lower()
    if not r:
        return "(none)"
    for k in ("timeout", "syntaxerror", "no code", "empty", "wrong answer", "assertion",
              "nameerror", "typeerror", "indexerror", "valueerror", "recursion",
              "memory", "indentation", "runtime", "attributeerror", "keyerror"):
        if k in r:
            return k
    return r.split(":")[0][:38]


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else 0


print("=== FAILURE REASON CENSUS ===")
for t, _ in CELLS:
    c = collections.Counter(bucket(d.get("reason")) for d in A[t].values() if not d.get("passed"))
    print("  %-6s n_fail=%2d  %s" % (t, sum(c.values()), dict(c.most_common())))

print("\n=== LENGTH: PASS vs FAIL (completion_tokens) ===")
for t, _ in CELLS:
    p = [d["completion_tokens"] for d in A[t].values() if d.get("passed") and isinstance(d.get("completion_tokens"), int)]
    f = [d["completion_tokens"] for d in A[t].values() if not d.get("passed") and isinstance(d.get("completion_tokens"), int)]
    print("  %-6s PASS n=%2d p50=%6d | FAIL n=%2d p50=%6d min=%5d" % (t, len(p), med(p), len(f), med(f), min(f) if f else 0))

print("\n=== structural: does the completion contain code at all? ===")
for t, _ in CELLS:
    nocode = [k for k, d in A[t].items() if not CODE.search(d.get("completion") or "")]
    tiny = [k for k, d in A[t].items() if (d.get("completion_tokens") or 0) < 2000]
    print("  %-6s no-code=%2d  under-2000-tok=%2d" % (t, len(nocode), len(tiny)))

print("\n=== DISCORDANT: armJ PASS, gepo2 FAIL ===")
disc = sorted(k for k in A["armJ"] if A["armJ"][k].get("passed") and not A["gepo2"][k].get("passed"))
print("  n=%d" % len(disc))
gd, ad = [], []
for k in disc:
    g, a = A["gepo2"][k], A["armJ"][k]
    gd.append(g.get("completion_tokens") or 0); ad.append(a.get("completion_tokens") or 0)
    print("   %-24s gepo2 tok=%6s fin=%-6s | armJ tok=%6s | %s"
          % (k, g.get("completion_tokens"), g.get("finish_reason"), a.get("completion_tokens"),
             (g.get("reason") or "")[:46]))
if disc:
    print("  -> on these, gepo2 p50=%d vs armJ p50=%d  (gepo2 shorter on %d/%d)"
          % (med(gd), med(ad), sum(1 for i in range(len(gd)) if gd[i] < ad[i]), len(gd)))

print("\n=== reverse DISCORDANT: gepo2 PASS, armJ FAIL ===")
rev = sorted(k for k in A["armJ"] if A["gepo2"][k].get("passed") and not A["armJ"][k].get("passed"))
print("  n=%d  %s" % (len(rev), rev))
