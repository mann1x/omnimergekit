import json, sys, os, glob
root = sys.argv[1]
marker = sys.argv[2] if len(sys.argv) > 2 else None  # path segment that precedes the bench name
rows = []
for f in sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)):
    try:
        d = json.load(open(f))
    except Exception as e:
        continue
    # bench name = the directory just under root, or after marker
    rel = os.path.relpath(f, root)
    parts = rel.split(os.sep)
    bench = parts[0]
    score = d.get("score")
    sc = round(score * 100, 2) if isinstance(score, (int, float)) else None
    metric = d.get("metric"); filt = d.get("filter")
    samp = (d.get("sampler") or {}).get("name", "?")
    rows.append((bench, sc, f"{metric},{filt}", samp))
w = max((len(r[0]) for r in rows), default=10)
for b, sc, mf, samp in rows:
    scs = f"{sc:>7}" if sc is not None else "   None"
    print(f"{b:<{w}}  {scs}  {mf:<22} sampler={samp}")
