"""Scan every LCB sweep cell for cache rows that are cached FAILURES, not generations.

Killing an eval mid-cell makes every still-pending request fail instantly against the
dead server, and the lcb shim caches those failures as responses: empty completion,
completion_tokens None, finish_reason None, gen_secs ~0. A later resume reads them as
finished work and scores them as wrong answers -- producing a plausible-looking but
entirely fictitious score (b12_msg/lcb_v6_77q: 0.1558 from 77 "problems" in 93 seconds,
62 of them null).

A score is only trustworthy if every row carries a real generation. This reports, per
cell, the null-row count so a contaminated cell can be purged and re-run.
"""
import glob
import json
import os
import pickle
import sqlite3
import sys

ROOTS = [("/mnt/sdc/lcb_v6_budget_probe", ["lcb_v6_77q", "lcb_v6_plus_73q"]),
         ("/mnt/sdc/lcb_budget_probe", ["lcb_hard_77"])]

bad = []
print("  %-26s %-16s %6s %6s %7s %9s  %s" %
      ("cell", "template", "rows", "null", "real", "score", "verdict"))
for root, tpls in ROOTS:
    for cell in sorted(os.listdir(root)):
        for tpl in tpls:
            d = "%s/%s/%s/%s" % (root, cell, tpl, cell)
            dbs = glob.glob(d + "/sqlite_cache/*.db")
            if not dbs:
                continue
            try:
                rs = [pickle.loads(v) for (v,) in
                      sqlite3.connect(dbs[0]).execute("select value from responses")]
            except Exception as e:
                print("  %-26s %-16s  (unreadable: %s)" % (cell, tpl, e))
                continue
            null = [r for r in rs if not (r.get("completion") or "").strip()]
            sf = d + "/summary.json"
            score = json.load(open(sf))["score"] if os.path.exists(sf) else None
            verdict = "OK" if not null else "CONTAMINATED — purge and re-run"
            if null:
                bad.append((root, cell, tpl, len(null), len(rs)))
            print("  %-26s %-16s %6d %6d %7d %9s  %s"
                  % (cell, tpl, len(rs), len(null), len(rs) - len(null),
                     ("%.4f" % score) if score is not None else "-", verdict))

print()
if bad:
    print("  %d contaminated cell(s). Their scores are FICTITIOUS and must not be reported:" % len(bad))
    for root, cell, tpl, n, tot in bad:
        print("    %s/%s  %d/%d rows are cached failures" % (cell, tpl, n, tot))
    sys.exit(1)
print("  every cell carries real generations in every row")
