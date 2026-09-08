import json, glob, os
for s in sorted(glob.glob("/srv/ml/eval_results_canon/*/*/summary.json")):
    j = json.load(open(s))
    bench = s.split(os.sep)[-3]; cell = s.split(os.sep)[-2]
    samp = (j.get("sampler") or {}).get("name")
    print(f"{bench:22s} {cell:26s} score={j.get('score')} metric={j.get('metric')} sampler={samp}")
    sc = j.get("scores")
    if isinstance(sc, dict):
        print("      per-lang:", {k: round(v,4) if isinstance(v,float) else v for k,v in sc.items()})
