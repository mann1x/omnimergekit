import json, glob, os, collections
R="/srv/ml/eval_results/qwen_suite/multipl_e_100"
ARMS={"pub":"qwencodermpe_q6k","armJ":"qwenhybridp24_q6k"}
def load(a,lang):
    o={}
    for f in glob.glob("%s/%s/results/%s/*.results.json"%(R,ARMS[a],lang)):
        if f.endswith("_summary.json"): continue
        d=json.load(open(f)); r0=(d.get("results") or [{}])[0]
        o[d.get("name") or os.path.basename(f)]=(str(r0.get("status")), r0.get("program") or "")
    return o

print("BRACE BALANCE (open - close). Negative = MORE closing braces than opening =")
print("the model emitted a terminal `}` that MultiPL-E's template also appends.")
print("Truncation would instead be POSITIVE (unclosed).\n")
for lang in ("humaneval-java","humaneval-rs","humaneval-js"):
    for a in ("pub","armJ"):
        P=load(a,lang)
        syn=collections.Counter(); ok=collections.Counter()
        for k,(st,p) in P.items():
            if not p: continue
            d=p.count("{")-p.count("}")
            (syn if st=="SyntaxError" else ok)[d]+=1
        print("  %-16s %-5s SyntaxError balance=%s   |   OK balance=%s"
              % (lang,a,dict(sorted(syn.items())),dict(sorted(ok.items()))))

print("\nDECISIVE: java problems where armJ SyntaxErrors and pub passes — is armJ's")
print("program otherwise the SAME shape, differing only by the extra brace?")
PJ,JJ=load("pub","humaneval-java"),load("armJ","humaneval-java")
po=[k for k in PJ if PJ[k][0]=="OK" and JJ.get(k,("",""))[0]=="SyntaxError"]
overclose=sum(1 for k in po if JJ[k][1].count("{")-JJ[k][1].count("}") < 0)
print("  armJ-fails/pub-passes: %d   of which armJ program is OVER-CLOSED: %d (%.0f%%)"
      % (len(po),overclose,100*overclose/max(len(po),1)))
print("  pub's programs on those same %d problems, balance: %s"
      % (len(po), dict(collections.Counter(PJ[k][1].count("{")-PJ[k][1].count("}") for k in po))))
