import json, sys, re
F = sys.argv[1]
n=0; correct=0; wrong_marker=0; wrong_nomarker=0
wm_len=[]; wnm_len=[]; ex=[]
# GPQA cot: answer emitted like "answer is (A)" / "The correct answer is B"
marker_re = re.compile(r"answer is[:\s]*\(?([A-D])\)?", re.IGNORECASE)
for line in open(F):
    line=line.strip()
    if not line: continue
    d=json.loads(line)
    n+=1
    em=d.get("exact_match")
    if em is None and isinstance(d.get("metrics"),dict): em=d["metrics"].get("exact_match")
    em=int(round(float(em))) if em is not None else 0
    if em==1: correct+=1; continue
    def flat(x):
        if isinstance(x,str): return x
        if isinstance(x,list): return " ".join(flat(i) for i in x)
        return str(x)
    txt=flat(d.get("resps") or d.get("filtered_resps") or [])
    L=len(txt)
    if marker_re.search(txt): wrong_marker+=1; wm_len.append(L)
    else:
        wrong_nomarker+=1; wnm_len.append(L)
        if len(ex)<3: ex.append((d.get("doc_id"),L,txt[-150:]))
def pct(a,b): return f"{100.0*a/b:.2f}%" if b else "n/a"
def st(name,xs):
    if not xs: print(f"  {name}: none"); return
    xs=sorted(xs);k=len(xs);print(f"  {name}: n={k} min={xs[0]} p50={xs[k//2]} max={xs[-1]} chars")
print(f"total={n} correct={correct} ({pct(correct,n)}) wrong={n-correct}")
print(f"  wrong WITH answer-marker (genuine wrong pick): {wrong_marker}")
print(f"  wrong WITHOUT marker (no clean answer):        {wrong_nomarker}")
st("wrong-with-marker len",wm_len); st("wrong-no-marker len",wnm_len)
print(f"UPPER-BOUND if every no-marker recoverable: {pct(correct+wrong_nomarker,n)}")
for did,L,tail in ex: print(f"[{did}] len={L} ...{tail!r}")
