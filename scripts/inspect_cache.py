import sqlite3, pickle, sys, statistics
db = sys.argv[1]
con = sqlite3.connect(db)
rows = con.execute("SELECT key, value FROM unnamed").fetchall()
print(f"total_rows={len(rows)}")
lengths = []
empties = 0
saturated = 0
finish_reasons = {}
for k, v in rows:
    try:
        obj = pickle.loads(v)
        # obj is typically list of strings (the generation)
        if isinstance(obj, list):
            for s in obj:
                if not s:
                    empties += 1
                    lengths.append(0)
                    continue
                lengths.append(len(s))
                if len(s) >= 16384*3.5:
                    saturated += 1
        else:
            s = str(obj)
            lengths.append(len(s))
    except Exception as e:
        print(f"decode err: {e}")
if lengths:
    lengths.sort()
    p50 = lengths[len(lengths)//2]
    p10 = lengths[len(lengths)//10] if len(lengths)>=10 else lengths[0]
    p90 = lengths[(len(lengths)*9)//10] if len(lengths)>=10 else lengths[-1]
    print(f"p10={p10}  p50={p50}  p90={p90}  min={min(lengths)}  max={max(lengths)}")
    print(f"empty={empties}  saturated_>57k={saturated}  responses={len(lengths)}")
    # print first 2 sample heads
    samples = sorted(rows, key=lambda r: len(r[1]))[:2] + sorted(rows, key=lambda r: -len(r[1]))[:1]
    for i,(k,v) in enumerate(samples):
        try:
            o = pickle.loads(v)
            s = o[0] if isinstance(o, list) and o else str(o)
            head = s[:200].replace('\n','\\n')
            print(f"---sample {i+1} (len={len(s)})---")
            print(head)
        except: pass
