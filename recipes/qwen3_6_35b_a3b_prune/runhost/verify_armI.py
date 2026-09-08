#!/usr/bin/env python
"""armI's 48k LCB cell claims 57/77 -- +10 over armD, from the arm that scored WORST on MPE.

That contradiction is exactly the shape of a scoring artefact, so check the samples before
believing the number (the standing rule: an anomalous pass rate may be the model OR the
scorer). Also ask whether armI's 57 CONTAINS armD's 47 or merely overlaps it -- a superset
is a capability gain, a reshuffle is a different model, not a better one.
"""
import json
import os

R = "/srv/ml/eval_results/ream_arms/lcb_v6_77q_48k"
ARMS = {"armI": "lcb48k_armI_hybrid_p12", "armD": "lcb48k_armD_ourssal_nomerge"}


def rows(name):
    p = os.path.join(R, name, "lcb_result.samples.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def field(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return None


for tag, name in ARMS.items():
    rs = rows(name)
    if not rs:
        print("%s: NO samples file" % tag)
        continue
    n = len(rs)
    empty = short = fenced = 0
    passes = 0
    for d in rs:
        c = field(d, "completion", "response", "output", "generation", "text") or ""
        if isinstance(c, list):
            c = " ".join(map(str, c))
        c = str(c)
        if not c.strip():
            empty += 1
        elif len(c.strip()) < 5:
            short += 1
        if "```" in c:
            fenced += 1
        p = field(d, "pass", "passed", "correct")
        if p is None:
            p = field(d, "score")
        if p:
            passes += 1
    print("%s: n=%d  passes=%d (%.4f)  empty=%d  <5char=%d  has_fence=%d"
          % (tag, n, passes, passes / n if n else 0, empty, short, fenced))
    print("     keys: %s" % sorted(rs[0].keys())[:14])

# --- superset or reshuffle? ---
def solved(name):
    p = os.path.join(R, name, "lcb_result.json")
    if not os.path.exists(p):
        return {}
    res = json.load(open(p))
    it = res
    if isinstance(res, dict):
        for k in ("results", "per_problem", "problems", "detail"):
            if isinstance(res.get(k), (list, dict)):
                it = res[k]
                break
    out = {}
    if isinstance(it, dict):
        for k, v in it.items():
            out[k] = bool(v if isinstance(v, bool)
                          else (v.get("pass") if isinstance(v, dict) else False))
    elif isinstance(it, list):
        for v in it:
            if isinstance(v, dict):
                tid = v.get("task_id") or v.get("id")
                for f in ("pass", "passed", "correct", "pass@1", "score"):
                    if f in v and tid:
                        out[tid] = bool(v[f])
                        break
    return out


I, D = solved(ARMS["armI"]), solved(ARMS["armD"])
common = set(I) & set(D)
if common:
    both = [t for t in common if I[t] and D[t]]
    i_only = sorted(t for t in common if I[t] and not D[t])
    d_only = sorted(t for t in common if D[t] and not I[t])
    print("\narmI vs armD on the SAME 48k basis (%d shared task_ids):" % len(common))
    print("  both        : %d" % len(both))
    print("  armI-only   : %d %s" % (len(i_only), i_only[:12]))
    print("  armD-only   : %d %s" % (len(d_only), d_only[:12]))
    print("  neither     : %d" % sum(1 for t in common if not I[t] and not D[t]))
    if not d_only:
        print("\n  => armI is a STRICT SUPERSET of armD. Real capability gain, not a trade.")
    else:
        print("\n  => TRADE: armI wins %d, loses %d. Net %+d."
              % (len(i_only), len(d_only), len(i_only) - len(d_only)))
