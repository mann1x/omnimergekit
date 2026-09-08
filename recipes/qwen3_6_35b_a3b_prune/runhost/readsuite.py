import json, glob, os
rows=[]
for s in sorted(glob.glob("/srv/ml/eval_results*/**/summary.json", recursive=True)):
    try: j=json.load(open(s))
    except Exception: continue
    parts=s.split(os.sep); bench=parts[-3]; cell=parts[-2]; root=parts[3] if len(parts)>3 else "?"
    if not any(k in cell.lower() for k in ("coderx","armj","184e","256e","pub","base")): continue
    samp=(j.get("sampler") or {}).get("name")
    rows.append((bench,cell,root,j.get("score"),samp))
for r in sorted(rows):
    print(f"{r[0]:24s} {r[1]:28s} {r[2]:22s} score={r[3]} sampler={r[4]}")
