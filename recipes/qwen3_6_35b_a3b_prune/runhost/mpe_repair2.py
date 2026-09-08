# DIAGNOSTIC ONLY -- scratch dir; official multipl_e_100 cell untouched.
import json, glob, os, re, subprocess, collections
R="/srv/ml/eval_results/qwen_suite/multipl_e_100"
W="/mnt/sdc/ream-work/mpe_java_repair2"; os.makedirs(W, exist_ok=True)
ARMS={"pub":"qwencodermpe_q6k","armJ":"qwenhybridp24_q6k"}
def load(a):
    o={}
    for f in glob.glob("%s/%s/results/humaneval-java/*.results.json"%(R,ARMS[a])):
        if f.endswith("_summary.json"): continue
        d=json.load(open(f)); r0=(d.get("results") or [{}])[0]
        o[d.get("name") or os.path.basename(f)]=(str(r0.get("status")), r0.get("program") or "")
    return o
J=load("armJ"); P=load("pub")

def run(src, tag):
    m=re.search(r'(?:public\s+)?(?:final\s+)?class\s+(\w+)', src)
    cls=m.group(1) if m else "Problem"
    d=os.path.join(W,tag); os.makedirs(d, exist_ok=True)
    p=os.path.join(d, cls+".java"); open(p,"w").write(src)
    try:
        r=subprocess.run(["java",p],capture_output=True,timeout=45,cwd=d)
        return r.returncode, r.stderr.decode()[:160]
    except subprocess.TimeoutExpired: return 124,"TIMEOUT"
    except Exception as e: return 125,str(e)[:160]

def repair(src):
    """Drop the lone `}` line that sits immediately before the harness-appended main."""
    lines=src.split("\n")
    mi=next((i for i,l in enumerate(lines) if "static void main" in l), None)
    if mi is None: return None
    j=mi-1
    while j>=0 and lines[j].strip()=="": j-=1
    if j>=0 and lines[j].strip()=="}":
        return "\n".join(lines[:j]+lines[j+1:])
    return None

targets=[k for k in J if J[k][0]=="SyntaxError" and J[k][1].count("{")-J[k][1].count("}")==-1]
ok=0; fail=0; norep=0; rec=[]; still=[]
for k in sorted(targets):
    r=repair(J[k][1])
    if r is None: norep+=1; continue
    rc,err=run(r,"rep")
    if rc==0: ok+=1; rec.append(k)
    else: fail+=1; still.append((k,err.strip().splitlines()[:1]))

print("armJ java, SyntaxError AND over-closed by one: %d" % len(targets))
print("  no lone `}` before main (repair N/A): %d" % norep)
print("  RECOVERED after removing that one brace: %d" % ok)
print("  still failing (real defect):            %d" % fail)
print("\nIMPLICATION for the java cell:")
print("  scored  armJ = 67/100 (0.670)  vs pub 78/100 (0.780)   = -11")
print("  armJ + brace repair = %d/100 (%.3f)  =>  net vs pub = %+d"
      % (67+ok,(67+ok)/100, (67+ok)-78))
if still:
    print("\n  still-failing detail:")
    for k,e in still[:10]: print("    %-40s %s" % (k,e))

# CONTROL: does pub have the same defect class, and does the same repair help it?
pt=[k for k in P if P[k][0]=="SyntaxError" and P[k][1].count("{")-P[k][1].count("}")==-1]
pok=0
for k in pt:
    r=repair(P[k][1])
    if r is not None and run(r,"repp")[0]==0: pok+=1
print("\nCONTROL pub: same defect on %d problems; repair recovers %d -> pub would be %d/100"
      % (len(pt),pok,78+pok))
print("SO THE LIKE-FOR-LIKE REPAIRED COMPARISON IS: armJ %d vs pub %d = %+d"
      % (67+ok, 78+pok, (67+ok)-(78+pok)))
