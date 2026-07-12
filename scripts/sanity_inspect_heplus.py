#!/usr/bin/env python3
# sanity_inspect_heplus.py — mandated lm-eval sanity check for the STD16 HE+ re-run.
# For each tier: record count, completion length dist, empty/short/fence counts, pass@1
# reconciliation vs summary.json, and 2 real example completions. Catches the
# "fast == empty/fenced/cached" trap. Usage: sanity_inspect_heplus.py [TIER ...]
import glob
import json
import os
import sys

WORK = "/mnt/sdc/ml/std16_gate/he_mpe"
TP = {"Q8_0": "08", "Q6_K_L": "09", "Q6_K": "09", "Q5_K_L": "09", "Q5_K_M": "09",
      "Q4_K_L": "08", "Q4_K_M": "09", "Q4_K_S": "09", "IQ4_NL": "09", "IQ4_XS": "09",
      "Q3_K_L": "08", "Q3_K_M": "09", "CD-Q2_K": "09"}
tiers = sys.argv[1:] or ["Q8_0", "Q6_K", "Q5_K_L"]


def comp(r):
    fr = r.get("filtered_resps") or r.get("resps") or []
    if isinstance(fr, list) and fr:
        x = fr[0]
        while isinstance(x, list) and x:
            x = x[0]
        return x if isinstance(x, str) else str(x)
    return ""


def passed(r):
    for k in ("pass@1", "passed", "exact_match", "acc"):
        if k in r:
            v = r[k]
            try:
                return float(v[0] if isinstance(v, list) else v)
            except Exception:
                return None
    return None


for T in tiers:
    B = "humanevalplus_full_minprep" + TP[T]
    d = os.path.join(WORK, "results", T, B)
    sf = sorted(glob.glob(os.path.join(d, "**", "samples*.jsonl"), recursive=True),
                key=os.path.getsize, reverse=True)
    print("\n===== %s (%s) =====" % (T, B))
    summ = sorted(glob.glob(os.path.join(d, "**", "summary.json"), recursive=True), key=len)
    if summ:
        j = json.load(open(summ[0]))
        print("  summary.score=%s metric=%s filter=%s" % (j.get("score"), j.get("metric"), j.get("filter")))
    if not sf:
        print("  NO samples file yet")
        continue
    print("  samples:", sf[0], "(%d bytes)" % os.path.getsize(sf[0]))
    rows = [json.loads(line) for line in open(sf[0]) if line.strip()]
    comps = [comp(r) for r in rows]
    lens = sorted(len(c) for c in comps)
    n = len(comps)
    empty = sum(1 for c in comps if len(c.strip()) == 0)
    short = sum(1 for c in comps if len(c.strip()) < 5)
    fence = sum(1 for c in comps if "```" in c)

    def pq(q):
        return lens[min(n - 1, int(q * n))] if n else 0
    print("  n_records=%d  (expect 164)" % n)
    print("  completion chars: p10=%d p50=%d p90=%d max=%d" % (pq(.1), pq(.5), pq(.9), lens[-1] if lens else 0))
    print("  empty=%d  short(<5)=%d  fenced(```)=%d" % (empty, short, fence))
    ps = [passed(r) for r in rows]
    psv = [x for x in ps if x is not None]
    if psv:
        print("  pass@1 from samples: mean=%.4f  n_scored=%d  n_pass=%d" %
              (sum(psv) / len(psv), len(psv), sum(1 for x in psv if x >= 0.5)))
    else:
        print("  pass@1: no per-sample score key found; keys=%s" % (list(rows[0].keys())[:12] if rows else []))
    for r in rows[:2]:
        tid = r.get("doc", {}).get("task_id") or r.get("doc_id", "?")
        print("  --- task=%s passed=%s" % (tid, passed(r)))
        print("     completion[:280]=%r" % (comp(r)[:280],))
