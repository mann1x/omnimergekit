import json, sys, re
F = sys.argv[1]
n=0; correct=0
wrong_with_marker=0   # emitted "the best answer is X" but X != target -> genuine wrong pick
wrong_no_marker=0     # no marker found -> extraction miss (usually runaway/long)
wrong_marker_lens=[]; wrong_nomarker_lens=[]
marker_re = re.compile(r"the best answer is", re.IGNORECASE)
examples_nomarker=[]
for line in open(F):
    line=line.strip()
    if not line: continue
    d=json.loads(line)
    n+=1
    em=d.get("exact_match")
    if em is None and isinstance(d.get("metrics"),dict):
        em=d["metrics"].get("exact_match")
    em=int(round(float(em))) if em is not None else 0
    if em==1:
        correct+=1; continue
    # wrong: inspect raw response text
    resps=d.get("resps") or d.get("filtered_resps") or []
    txt=""
    # resps can be list[list[str]] or list[str]
    def flat(x):
        if isinstance(x,str): return x
        if isinstance(x,list): return " ".join(flat(i) for i in x)
        return str(x)
    txt=flat(resps)
    L=len(txt)
    if marker_re.search(txt):
        wrong_with_marker+=1; wrong_marker_lens.append(L)
    else:
        wrong_no_marker+=1; wrong_nomarker_lens.append(L)
        if len(examples_nomarker)<4:
            examples_nomarker.append((d.get("doc_id"), L, d.get("target"), txt[-180:]))
def pct(a,b): return f"{100.0*a/b:.2f}%" if b else "n/a"
print(f"total={n} correct={correct} ({pct(correct,n)}) wrong={n-correct}")
print(f"  wrong WITH 'best answer is' marker (genuine wrong pick): {wrong_with_marker}")
print(f"  wrong WITHOUT marker (extraction miss / runaway):        {wrong_no_marker}")
def stats(name,xs):
    if not xs: print(f"  {name}: none"); return
    xs=sorted(xs); k=len(xs)
    print(f"  {name}: n={k} min={xs[0]} p50={xs[k//2]} max={xs[-1]} chars")
stats("wrong-with-marker char-len", wrong_marker_lens)
stats("wrong-no-marker char-len", wrong_nomarker_lens)
# upper bound: if all no-marker were actually right, score would be:
ub = correct + wrong_no_marker
print(f"UPPER-BOUND score if every no-marker were a recoverable answer: {pct(ub,n)} (correct+{wrong_no_marker})")
print("--- no-marker examples (doc_id, len, target, last180chars) ---")
for did,L,tg,tail in examples_nomarker:
    print(f"[{did}] len={L} target={tg!r} ...{tail!r}")
