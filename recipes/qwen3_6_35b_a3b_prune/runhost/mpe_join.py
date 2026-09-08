import json, glob, os, collections
R="/srv/ml/eval_results/qwen_suite/multipl_e_100"
ARMS={"pub":"qwencodermpe_q6k","armJ":"qwenhybridp24_q6k"}
langs=sorted({os.path.basename(d) for a in ARMS.values()
              for d in glob.glob(R+"/"+a+"/results/*") if os.path.isdir(d)})
print("languages:", langs, "\n")

def cell(arm, lang):
    o={}
    for f in glob.glob("%s/%s/results/%s/*.results.json" % (R,ARMS[arm],lang)):
        if f.endswith("_summary.json"): continue
        d=json.load(open(f))
        name=d.get("name") or os.path.basename(f).replace(".results.json","")
        res=d.get("results") or []
        r0=res[0] if res else {}
        st=r0.get("status") or r0.get("exit_code")
        ok = (st=="OK") if isinstance(st,str) else (st==0)
        o[name]=(1 if ok else 0, str(st), (r0.get("stderr") or "")[:200])
    return o

tot=collections.Counter()
for lang in langs:
    P,J=cell("pub",lang),cell("armJ",lang)
    ids=sorted(set(P)&set(J))
    pp=sum(P[i][0] for i in ids); jj=sum(J[i][0] for i in ids)
    po=[i for i in ids if P[i][0] and not J[i][0]]
    jo=[i for i in ids if J[i][0] and not P[i][0]]
    tot["pub"]+=pp; tot["armJ"]+=jj; tot["n"]+=len(ids)
    print("%-16s n=%3d  pub %3d (%.3f)  armJ %3d (%.3f)  net=%+d   pub-only=%d armJ-only=%d"
          % (lang,len(ids),pp,pp/len(ids),jj,jj/len(ids),jj-pp,len(po),len(jo)))
    if po:
        print("   armJ FAILS where pub passes (%d):" % len(po))
        kinds=collections.Counter(J[i][1] for i in po)
        print("     armJ failure statuses:", dict(kinds))
        for i in po[:6]:
            print("       %-38s armJ=%s" % (i, J[i][1]))
    if jo:
        print("   armJ WINS where pub fails (%d): %s" % (len(jo), ", ".join(jo[:6])))
print("\nTOTAL n=%d  pub=%d (%.4f)  armJ=%d (%.4f)  net=%+d"
      % (tot["n"],tot["pub"],tot["pub"]/tot["n"],tot["armJ"],tot["armJ"]/tot["n"],tot["armJ"]-tot["pub"]))
