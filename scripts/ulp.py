import glob, torch, json
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
    if t[k] not in hs: hs[t[k]]=safe_open(t[k],"pt")
    return hs[t[k]].get_tensor(k)
k="model.language_model.layers.10.linear_attn.norm.weight"
a,b=get(ox,k),get(qb,k)
d=(a.float()-b.float())
nz=(d!=0)
print("tensor:",k,"numel",a.numel(),"dtype",a.dtype)
print("  elements differing: %d / %d (%.2f%%)" % (int(nz.sum()),a.numel(),100*nz.float().mean()))
i=nz.nonzero()[:5].flatten()
for j in i:
    av,bv=a[j].item(),b[j].item()
    ulp=torch.finfo(torch.bfloat16).eps*max(abs(bv),1e-9)
    print("    idx %-6d ox=%.8f qwen=%.8f  delta=%+.8f  ~%.2f ULP(bf16)" % (j,av,bv,av-bv,(av-bv)/ulp if ulp else 0))
print()
print("=== relative magnitude of the ONLY signal OxCoder carries ===")
tot=0.0; totn=0.0
for kk in ox:
    if "linear_attn.norm" in kk:
        aa,bb=get(ox,kk).float(),get(qb,kk).float()
        tot+=(aa-bb).norm().item()**2; totn+=bb.norm().item()**2
print("  ||OxCoder - Qwen|| / ||Qwen||  over the 24 differing tensors = %.6f%%" % (100*(tot**0.5)/(totn**0.5)))
print()
print("=== configs differ? ===")
c1=json.load(open(R+"/OxCoder-9B/config.json")); c2=json.load(open(R+"/Qwen3.5-9B/config.json"))
ks=set(c1)|set(c2)
for kk in sorted(ks):
    if c1.get(kk)!=c2.get(kk):
        s1,s2=str(c1.get(kk))[:60],str(c2.get(kk))[:60]
        print("  %-28s ox=%s | qwen=%s" % (kk,s1,s2))
