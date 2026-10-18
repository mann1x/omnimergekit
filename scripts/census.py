import glob, torch, collections
from safetensors import safe_open
R="/mnt/sdc/oxopus/models"
def load(d):
    t={}
    for f in sorted(glob.glob(d+"/*.safetensors")):
        with safe_open(f,"pt") as h:
            for k in h.keys(): t[k]=f
    return t
ox=load(R+"/OxCoder-9B"); qb=load(R+"/Qwen3.5-9B")
hs={}
def get(t,k):
    f=t[k]
    if f not in hs: hs[f]=safe_open(f,"pt")
    return hs[f].get_tensor(k)
common=sorted(k for k in ox if k in qb)
diff=[]; same=0
for k in common:
    a=get(ox,k); b=get(qb,k)
    if a.shape!=b.shape: diff.append((k,float("nan"))); continue
    if torch.equal(a,b): same+=1
    else: diff.append((k,(a.float()-b.float()).abs().max().item()))
print("OxCoder vs Qwen3.5-9B over %d common tensors:" % len(common))
print("  BIT-IDENTICAL : %d" % same)
print("  DIFFERENT     : %d" % len(diff))
if diff:
    print("  top differing tensors by max|delta|:")
    for k,v in sorted(diff,key=lambda x:-(x[1] if x[1]==x[1] else 0))[:15]:
        print("    %-66s %.6f" % (k[:66],v))
    pre=collections.Counter()
    for k,_ in diff:
        p=k.split(".")
        pre[".".join(p[:3])+("."+p[4] if len(p)>4 else "")]+=1
    print("  differing by group:")
    for p,c in pre.most_common(12): print("    %-50s %d" % (p,c))
