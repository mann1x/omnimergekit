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

print("THE DISCRIMINATOR — for each problem armJ passed on HE but failed on HE+,")
print("is the HE+ completion the SAME code (=> extra tests are the cause, a real")
print("robustness gap) or DIFFERENT code (=> just a different sampled draw)?\n")
for arm,probs in (("armJ",["HumanEval/124","HumanEval/126","HumanEval/130","HumanEval/154","HumanEval/76"]),):
    for i in probs:
        same = D[("HE",arm)][i][1] == D[("HE+",arm)][i][1]
        print("  %-16s armJ HE=%d HE+=%d  completion %s | pub HE+=%d"
              % (i, D[("HE",arm)][i][0], D[("HE+",arm)][i][0],
                 "IDENTICAL -> extra tests" if same else "DIFFERENT -> new draw",
                 D[("HE+","pub")][i][0]))

print("\nSame question for armJ's 3 stable wins:")
for i in ["HumanEval/116","HumanEval/163","HumanEval/67"]:
    print("  %-16s armJ HE=%d HE+=%d (same code across draws: %s) | pub HE=%d HE+=%d"
          % (i, D[("HE","armJ")][i][0], D[("HE+","armJ")][i][0],
             D[("HE","armJ")][i][1]==D[("HE+","armJ")][i][1],
             D[("HE","pub")][i][0], D[("HE+","pub")][i][0]))

print("\nDRAW STABILITY OVERALL (how much of any gap is resampling, not model):")
for arm in ("pub","armJ"):
    ids=set(D[("HE",arm)])
    diff=[i for i in ids if D[("HE",arm)][i][1]!=D[("HE+",arm)][i][1]]
    flip=[i for i in diff if (D[("HE",arm)][i][0]>=1)!=(D[("HE+",arm)][i][0]>=1)]
    print("  %-5s completions differing between the two draws: %3d/164; of those, verdict flips: %d"
          % (arm,len(diff),len(flip)))
