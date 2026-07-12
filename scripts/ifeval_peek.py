import json, glob, sys
BASE="/srv/ml/eval_results_tracks_2_3/ifeval_100"
def get_resp(d):
    fr=d.get("filtered_resps")
    if fr:
        x=fr[0]
        if isinstance(x,list): x=x[0] if x else ""
        return x or ""
    return ""
def prompt_of(d):
    a=d.get("doc",{})
    return a.get("prompt","")[:240]
dirn, mode = sys.argv[1], sys.argv[2]
f=sorted(glob.glob(f"{BASE}/{dirn}/lm_eval_out/*/samples_*.jsonl"))[-1]
rows=[json.loads(l) for l in open(f) if l.strip()]
def looped(t):
    w=t.split()
    if len(w)>=24:
        from collections import Counter
        g=[" ".join(w[i:i+8]) for i in range(len(w)-7)]
        c=Counter(g)
        return c and max(c.values())>=3
    return False
if mode=="runaway":
    cand=sorted(rows, key=lambda d: -len(get_resp(d)))[:2]
elif mode=="loop":
    cand=[d for d in rows if looped(get_resp(d))][:2]
for d in cand:
    t=get_resp(d); L=len(t)
    print(f"\n===== doc_id={d.get('doc_id')} len={L} pass_strict={d.get('prompt_level_strict_acc')} =====")
    print("PROMPT:", prompt_of(d).replace(chr(10)," "))
    print("--- answer head (400c):", repr(t[:400]))
    print("--- answer tail (400c):", repr(t[-400:]))
