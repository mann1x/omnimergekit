#!/usr/bin/env python
"""Is the armD 48k cell the SAME measurement as its 32k cell, or a coincidence of aggregates?

Three aggregates matching (score, len-term count, p50) is not proof the columns agree: a
reshuffle of equal size prints the same numbers. Compare the PER-PROBLEM solve sets, and
prove the 48k run actually generated rather than re-serving cached 32k completions.
"""
import json
import os
import sqlite3

R = "/srv/ml/eval_results/ream_arms"
CELLS = [
    ("32k", "lcb_v6_77q", "lcb_base256e"),
    ("48k", "lcb_v6_77q_48k", "lcb48k_base256e"),
]


def load(tmpl, name):
    d = os.path.join(R, tmpl, name)
    s = json.load(open(os.path.join(d, "summary.json")))
    res = None
    for cand in ("lcb_result.json", "results.json"):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            res = json.load(open(p))
            break
    return d, s, res


def solved(res):
    """-> {task_id: bool}. LCB result shapes vary; probe the common ones."""
    if res is None:
        return {}
    rows = res
    if isinstance(res, dict):
        for k in ("results", "per_problem", "problems", "detail"):
            if isinstance(res.get(k), (list, dict)):
                rows = res[k]
                break
    out = {}
    if isinstance(rows, dict):
        for k, v in rows.items():
            if isinstance(v, bool):
                out[k] = v
            elif isinstance(v, dict):
                for f in ("pass", "passed", "correct", "pass@1", "score"):
                    if f in v:
                        out[k] = bool(v[f])
                        break
    elif isinstance(rows, list):
        for v in rows:
            if not isinstance(v, dict):
                continue
            tid = v.get("task_id") or v.get("id") or v.get("question_id")
            for f in ("pass", "passed", "correct", "pass@1", "score"):
                if f in v and tid is not None:
                    out[tid] = bool(v[f])
                    break
    return out


info = {}
for tag, tmpl, name in CELLS:
    d, s, res = load(tmpl, name)
    ts = s.get("token_stats") or {}
    c = ts.get("completion_tokens") or {}
    fr = ts.get("finish_reasons") or {}
    sv = solved(res)
    info[tag] = sv
    # prove the cache is this basis's own, not the other ceiling's
    cdir = os.path.join(d, "sqlite_cache")
    dbs = sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []
    nrows, mtime = None, None
    if dbs:
        p = os.path.join(cdir, dbs[0])
        mtime = os.path.getmtime(p)
        try:
            con = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
            t = [r[0] for r in con.execute(
                "select name from sqlite_master where type='table'")]
            if t:
                nrows = con.execute("select count(*) from %s" % t[0]).fetchone()[0]
            con.close()
        except Exception as e:
            nrows = "err:%s" % e
    import datetime
    mt = datetime.datetime.utcfromtimestamp(mtime).strftime("%m-%d %H:%M") if mtime else "-"
    print("%-4s score=%.4f  n=%s  fr=%s  p50=%s max=%s | cache=%s rows=%s mtime=%sUTC | scored=%d"
          % (tag, s.get("score"), ts.get("n"), dict(fr), c.get("p50"), c.get("max"),
             dbs[0] if dbs else "NONE", nrows, mt, len(sv)))

a, b = info["32k"], info["48k"]
common = set(a) & set(b)
if not common:
    print("\nNO per-problem overlap recoverable -- cannot rule out a reshuffle from aggregates alone.")
else:
    only_a = sorted(t for t in common if a[t] and not b[t])
    only_b = sorted(t for t in common if b[t] and not a[t])
    print("\nper-problem over %d shared task_ids:" % len(common))
    print("  solved in BOTH      : %d" % sum(1 for t in common if a[t] and b[t]))
    print("  32k-only (lost @48k): %d %s" % (len(only_a), only_a[:8]))
    print("  48k-only (gained)   : %d %s" % (len(only_b), only_b[:8]))
    print("  failed in BOTH      : %d" % sum(1 for t in common if not a[t] and not b[t]))
    if not only_a and not only_b:
        print("\n  => IDENTICAL SOLVE SETS. Raising the ceiling changed nothing for this arm;"
              "\n     the 32k ceiling was never binding on armD.")
    else:
        print("\n  => RESHUFFLE: equal aggregates, different problems. Do NOT call these equivalent.")
