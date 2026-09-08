import json, glob, os, collections
R="/srv/ml/eval_results/qwen_suite/multipl_e_100"
print("Blast radius of the chat_to_body body-only fallback (completion brace balance < 0).")
print("A negative balance in the COMPLETION means an opening line was eaten.\n")
print("%-24s %-16s %5s %5s %6s %7s" % ("arm","language","n","neg","fail@neg","score"))
tot=collections.Counter()
for arm in sorted(os.listdir(R)):
    for lang in ("humaneval-java","humaneval-js","humaneval-rs"):
        gd="%s/%s/generations/%s"%(R,arm,lang); rd="%s/%s/results/%s"%(R,arm,lang)
        if not os.path.isdir(gd): continue
        neg=[]; n=0
        for f in glob.glob(gd+"/*.json"):
            d=json.load(open(f)); c=(d.get("completions") or [""])[0]
            n+=1
            if c.count("{")-c.count("}") < 0: neg.append(d["name"])
        if not n: continue
        failneg=0; ok=0; tt=0
        for f in glob.glob(rd+"/*.results.json"):
            if f.endswith("_summary.json"): continue
            d=json.load(open(f)); r0=(d.get("results") or [{}])[0]
            st=str(r0.get("status")); tt+=1
            if st=="OK": ok+=1
            if d.get("name") in neg and st!="OK": failneg+=1
        print("%-24s %-16s %5d %5d %8d %7s" % (arm,lang,n,len(neg),failneg,
              "%.3f"%(ok/tt) if tt else "-"))
        tot[(lang,"neg")]+=len(neg); tot[(lang,"failneg")]+=failneg; tot[(lang,"n")]+=n
print()
for lang in ("humaneval-java","humaneval-js","humaneval-rs"):
    if tot[(lang,"n")]:
        print("TOTAL %-16s cells=%d  eaten-line completions=%d  of which fail=%d"
              % (lang,tot[(lang,"n")],tot[(lang,"neg")],tot[(lang,"failneg")]))
