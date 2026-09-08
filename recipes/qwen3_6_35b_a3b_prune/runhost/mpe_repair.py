# DIAGNOSTIC ONLY. Writes to a scratch dir; the official multipl_e_100 cell is untouched.
# Question: are armJ's java SyntaxError programs correct code carrying ONE spurious `}`?
import json, glob, os, re, subprocess, collections
R="/srv/ml/eval_results/qwen_suite/multipl_e_100"
W="/mnt/sdc/ream-work/mpe_java_repair"; os.makedirs(W, exist_ok=True)
ARMS={"pub":"qwencodermpe_q6k","armJ":"qwenhybridp24_q6k"}

def load(a):
    o={}
    for f in glob.glob("%s/%s/results/humaneval-java/*.results.json"%(R,ARMS[a])):
        if f.endswith("_summary.json"): continue
        d=json.load(open(f)); r0=(d.get("results") or [{}])[0]
        o[d.get("name") or os.path.basename(f)]=(str(r0.get("status")), r0.get("program") or "")
    return o
P,J=load("pub"),load("armJ")

targets=[k for k in J if J[k][0]=="SyntaxError"
         and J[k][1].count("{")-J[k][1].count("}")==-1]
print("armJ java SyntaxError programs over-closed by exactly one brace: %d\n" % len(targets))

def run(src, tag):
    m=re.search(r'(?:public\s+)?(?:final\s+)?class\s+(\w+)', src)
    cls=m.group(1) if m else "Problem"
    d=os.path.join(W,tag); os.makedirs(d, exist_ok=True)
    p=os.path.join(d, cls+".java"); open(p,"w").write(src)
    try:
        r=subprocess.run(["java",p], capture_output=True, timeout=45, cwd=d)
        return r.returncode, (r.stderr.decode()[:150])
    except subprocess.TimeoutExpired: return 124,"TIMEOUT"
    except Exception as e: return 125,str(e)[:150]

def strip_last_brace(s):
    i=s.rfind("}")
    return s[:i]+s[i+1:] if i>=0 else s

res=collections.Counter(); fixed=[]; still=[]
for k in sorted(targets):
    src=J[k][1]
    rc0,_=run(src,"asis")
    rc1,err=run(strip_last_brace(src),"fixed")
    res[("asis_ok" if rc0==0 else "asis_fail")]+=1
    if rc1==0: res["fixed_ok"]+=1; fixed.append(k)
    else:      res["fixed_fail"]+=1; still.append((k,err.strip().splitlines()[:1]))

print("as-is (what MultiPL-E scored): %d pass / %d fail" % (res["asis_ok"],res["asis_fail"]))
print("after removing ONE spurious closing brace: %d PASS / %d still fail" % (res["fixed_ok"],res["fixed_fail"]))
print("\nRECOVERED (%d): %s" % (len(fixed), ", ".join(fixed)))
if still:
    print("\nSTILL FAILING (%d) -- these are real defects, not the brace:" % len(still))
    for k,e in still: print("   %-40s %s" % (k,e))
