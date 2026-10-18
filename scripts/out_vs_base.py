import glob, torch
from safetensors import safe_open
R="/mnt/sdc/oxopus/models"; OUT="/mnt/sdc/oxopus/OxOpusCoder-9B"
def load(d):
    t={}
    for f in sorted(glob.glob(d+"/*.safetensors")):
        with safe_open(f,"pt") as h:
            for k in h.keys(): t[k]=f
    return t
o=load(OUT); q=load(R+"/Qwopus3.5-9B-Coder")
hs={}
def get(t,k):
    if t[k] not in hs: hs[t[k]]=safe_open(t[k],"pt")
    return hs[t[k]].get_tensor(k)
same=0; diff=[]; total=0
for k in sorted(o):
    a,b=get(o,k),get(q,k); total+=a.numel()
    if torch.equal(a,b): same+=1
    else: diff.append((k,(a.float()-b.float()).abs().max().item(),a.numel()))
print("MERGED OUTPUT vs Qwopus (the base):")
print("  total params      : %.2f B" % (total/1e9))
print("  BIT-IDENTICAL     : %d / %d tensors" % (same,len(o)))
print("  changed           : %d tensors, %d params (%.6f%% of model)" % (
    len(diff), sum(d[2] for d in diff), 100*sum(d[2] for d in diff)/total))
for k,v,n in sorted(diff,key=lambda x:-x[1])[:6]:
    print("    %-60s max|d|=%.6f n=%d" % (k[:60],v,n))
