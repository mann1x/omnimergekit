import sqlite3, json
p = "/srv/ml/eval_results/ream_arms/multipl_e_100/a3b_gepo1/sqlite_cache/mpe_100_a3b_gepo1.db"
c = sqlite3.connect(p)
k, v = c.execute("select key,value from responses limit 1").fetchone()
print("KEY:", repr(k)[:300])
print("VAL len:", len(v))
try:
    d = json.loads(v)
    print("VAL is JSON:", list(d) if isinstance(d, dict) else type(d))
    if isinstance(d, dict):
        for kk in d:
            print("   %-18s %s" % (kk, repr(d[kk])[:170]))
except Exception as e:
    print("not json:", e)
    print(repr(v)[:400])
