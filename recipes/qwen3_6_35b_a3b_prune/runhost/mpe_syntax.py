import json, glob, os, collections
R="/srv/ml/eval_results/qwen_suite/multipl_e_100"
ARMS={"pub":"qwencodermpe_q6k","armJ":"qwenhybridp24_q6k"}
langs=["humaneval-java","humaneval-js","humaneval-rs"]

print("BASE RATE — status census per arm per language (the realized floor):")
St={}
for lang in langs:
    for arm,a in ARMS.items():
        c=collections.Counter(); prog={}
        for f in glob.glob("%s/%s/results/%s/*.results.json"%(R,a,lang)):
            if f.endswith("_summary.json"): continue
            d=json.load(open(f)); r0=(d.get("results") or [{}])[0]
            c[str(r0.get("status") or r0.get("exit_code"))]+=1
            prog[d.get("name") or os.path.basename(f)]=(str(r0.get("status")), r0.get("program") or d.get("program") or "")
        St[(lang,arm)]=prog
        print("  %-16s %-5s %s" % (lang,arm,dict(c)))

print("\nTRUNCATION CHECK — is the SyntaxError un-compilable code, or a generation that")
print("was CUT OFF mid-function? Unbalanced braces == truncated, not a syntax mistake.")
for lang in ("humaneval-java","humaneval-rs"):
    for arm in ("pub","armJ"):
        prog=St[(lang,arm)]
        syn=[k for k,v in prog.items() if v[0]=="SyntaxError"]
        unb=0; ends=collections.Counter()
        for k in syn:
            p=prog[k][1] or ""
            unb += 1 if p.count("{")!=p.count("}") else 0
            ends[(p.rstrip()[-1:] or "?")]+=1
        print("  %-16s %-5s SyntaxError=%2d  unbalanced-braces=%2d  last-char histogram=%s"
              % (lang,arm,len(syn),unb,dict(ends)))

print("\nSAMPLE — tail of one armJ java SyntaxError program:")
prog=St[("humaneval-java","armJ")]
for k,v in prog.items():
    if v[0]=="SyntaxError" and v[1]:
        p=v[1]
        print("  problem: %s   program chars=%d  braces {=%d }=%d"
              % (k,len(p),p.count("{"),p.count("}")))
        print("  ---- last 400 chars ----")
        print(p[-400:])
        break
