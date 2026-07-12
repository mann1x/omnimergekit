#!/usr/bin/env python3
"""Report response-length health of every v6-coder CD-* HE+/MPE eval cache."""
import sqlite3
import pickle
import glob


def dist(db):
    try:
        rows = sqlite3.connect(db).execute("select value from unnamed").fetchall()
    except Exception as e:
        return f"(err {e})"
    lens = sorted(len(str(pickle.loads(v))) for (v,) in rows if v)
    if not lens:
        return "(empty)"
    p50 = lens[len(lens) // 2]
    big = sum(1 for x in lens if x > 40000)
    return f"n={len(lens)} p50={p50} max={lens[-1]} >40k={big}"


hits = [d for d in glob.glob("/srv/ml/eval_results*/**/sqlite_cache/*rank0.db", recursive=True)
        if "CD" in d and "v6" in d.lower()]
print(f"found {len(hits)} v6-CD caches")
for db in sorted(hits):
    parts = db.split("/")
    print(f"  {parts[-4]}/{parts[-3]}: {dist(db)}")
