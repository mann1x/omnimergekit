#!/usr/bin/env python3
# Zero-GPU decoder for the killed IQ3_M HE+ run's sqlite response cache.
# Decides whether IQ3_M is genuinely degenerate (loops / malformed) or whether
# the eval just hit a few bad problems. Inspects the cached model outputs left
# behind by the 95/164 partial run. Prints, per cached response: length, a
# loop-signal (longest run of a repeated 12-char window), and a head/tail
# snippet. No re-run, no GPU.
import glob
import os
import pickle
import sqlite3
import sys


def loopiness(s, w=12):
    # crude repetition signal: max count of any length-w substring
    if len(s) < w * 3:
        return 1
    seen = {}
    best = 1
    for i in range(0, len(s) - w, 3):
        sub = s[i:i + w]
        seen[sub] = seen.get(sub, 0) + 1
        if seen[sub] > best:
            best = seen[sub]
    return best


def as_text(v):
    # cached value may be a dict (chat resp), list, or raw string
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("content", "text", "generated_text", "response"):
            if k in v and isinstance(v[k], str):
                return v[k]
        return repr(v)[:400]
    if isinstance(v, (list, tuple)) and v:
        return as_text(v[0])
    return repr(v)[:400]


root = sys.argv[1]
dbs = glob.glob(os.path.join(root, "**", "*.db"), recursive=True) + \
      glob.glob(os.path.join(root, "**", "*.sqlite"), recursive=True)
print("dbs found:", dbs)
for db in dbs:
    print("=" * 70)
    print("DB:", db)
    try:
        con = sqlite3.connect(db)
        tabs = [r[0] for r in con.execute(
            "select name from sqlite_master where type='table'")]
        print("tables:", tabs)
        for t in tabs:
            try:
                n = con.execute("select count(*) from '%s'" % t).fetchone()[0]
            except Exception as e:
                print("  table %s: count err %s" % (t, e))
                continue
            print("  table %s: %d rows" % (t, n))
            cols = [d[1] for d in con.execute("PRAGMA table_info('%s')" % t)]
            print("    cols:", cols)
            rows = con.execute("select * from '%s' limit 6" % t).fetchall()
            for ri, row in enumerate(rows):
                for cell in row:
                    if isinstance(cell, (bytes, bytearray)):
                        try:
                            v = pickle.loads(cell)
                        except Exception:
                            continue
                        txt = as_text(v)
                        lp = loopiness(txt)
                        head = txt[:160].replace("\n", "\\n")
                        tail = txt[-120:].replace("\n", "\\n")
                        print("    [r%d] len=%d loop=%d head=%s | tail=%s"
                              % (ri, len(txt), lp, head, tail))
                        break
        con.close()
    except Exception as e:
        print("  ERR", e)
