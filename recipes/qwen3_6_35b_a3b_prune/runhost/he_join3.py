import json, glob
R="/srv/ml/eval_results/qwen_suite"
C={("HE","pub"):R+"/humaneval_full_think/qwencodermpe_q6k",
   ("HE","armJ"):R+"/humaneval_full_think/qwenhybridp24_q6k",
   ("HE+","pub"):R+"/humanevalplus_full_think/qwencodermpe_q6k",
   ("HE+","armJ"):R+"/humanevalplus_full_think/qwenhybridp24_q6k"}
def load(d):
    f=sorted(glob.glob(d+"/lm_eval_out/*/samples_*.jsonl"))[0]; o={}
    for line in open(f):
        r=json.loads(line); tid=(r.get("doc") or {}).get("task_id")
        p=next((r[k] for k in r if k.startswith("pass@1")),None)
        rs=r.get("filtered_resps") or r.get("resps") or []
        t=rs[0] if rs and isinstance(rs[0],str) else (rs[0][0] if rs and isinstance(rs[0],(list,tuple)) and rs[0] else "")
        o[tid]=(float(p),t.strip())
    return o
D={k:load(v) for k,v in C.items()}
ids=sorted(D[("HE","pub")])

stable=[i for i in ids if D[("HE","pub")][i][1]==D[("HE+","pub")][i][1]
                      and D[("HE","armJ")][i][1]==D[("HE+","armJ")][i][1]]
print("DRAW-STABLE SUBSET: %d/164 problems where BOTH arms emitted identical code in" % len(stable))
print("both the HE and the HE+ run. On these, HE vs HE+ differs ONLY by the extra tests,")
print("so the comparison is not contaminated by resampling.\n")
for b in ("HE","HE+"):
    pp=sum(1 for i in stable if D[(b,"pub")][i][0]>=1)
    jj=sum(1 for i in stable if D[(b,"armJ")][i][0]>=1)
    po=[i for i in stable if D[(b,"pub")][i][0]>=1 and D[(b,"armJ")][i][0]<1]
    jo=[i for i in stable if D[(b,"pub")][i][0]<1 and D[(b,"armJ")][i][0]>=1]
    print("  %-3s  pub %3d/%d (%.4f)   armJ %3d/%d (%.4f)   armJ-only=%d pub-only=%d  net=%+d"
          % (b,pp,len(stable),pp/len(stable),jj,len(stable),jj/len(stable),len(jo),len(po),len(jo)-len(po)))
    if po: print("       pub-only: %s" % ", ".join(po))
    if jo: print("       armJ-only: %s" % ", ".join(jo))

print("\nEXTRA-TEST ATTRITION on the draw-stable subset (HE pass -> HE+ fail, same code):")
for arm in ("pub","armJ"):
    dn=[i for i in stable if D[("HE+",arm)][i][0]<1<=D[("HE",arm)][i][0]]
    print("  %-5s %d/%d = %.1f%%  %s" % (arm,len(dn),len(stable),100*len(dn)/len(stable),dn))
