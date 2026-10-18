"""SSM tensors: same VALUES permuted (conversion layout), or different content?"""
import json, glob, urllib.request, numpy as np, torch
from safetensors import safe_open
H=json.load(open("/mnt/sdc/oxopus/gguf_header.json")); URL,DS=H["url"],H["data_start"]
INFO={i["name"]:i for i in H["infos"]}; R="/mnt/sdc/oxopus/models"
def grab(n,nb=None):
    i=INFO[n]; esz=4 if i["type"]==0 else 2
    N=int(np.prod(i["dims"]))*esz; N=N if nb is None else min(nb,N); N-=N%esz
    s=DS+i["offset"]
    raw=urllib.request.urlopen(urllib.request.Request(URL,headers={"Range":f"bytes={s}-{s+N-1}"}),timeout=120).read()
    if i["type"]==0: return torch.from_numpy(np.frombuffer(raw,dtype="<f4").copy())
    return torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy()).view(torch.bfloat16)
def load(d):
    t={}
    for f in sorted(glob.glob(d+"/*.safetensors")):
        with safe_open(f,"pt") as h:
            for k in h.keys(): t[k]=f
    return t
QB,OX=load(R+"/Qwen3.5-9B"),load(R+"/OxCoder-9B"); hs={}
def get(t,k):
    if t[k] not in hs: hs[t[k]]=safe_open(t[k],"pt")
    return hs[t[k]].get_tensor(k)
L="model.language_model.layers"
def ms(a,b):  # multiset equality
    return torch.equal(torch.sort(a.flatten().float())[0], torch.sort(b.flatten().float())[0])
print("SSM tensors: is GGUF a PERMUTATION of the HF values? (sorted multiset compare)")
print("%-28s %-14s %-14s" % ("tensor","==sorted(Qwen)","==sorted(OxCoder)"))
for n in [0,2,12,20,30]:
    for gs,hs_ in [("ssm_dt.bias","linear_attn.dt_bias"),
                   ("ssm_conv1d.weight","linear_attn.conv1d.weight"),
                   ("ssm_out.weight","linear_attn.out_proj.weight")]:
        g=grab(f"blk.{n}.{gs}")
        q=get(QB,f"{L}.{n}.{hs_}"); o=get(OX,f"{L}.{n}.{hs_}")
        print("  blk.%-2d %-21s %-14s %-14s" % (n,gs,ms(g,q),ms(g,o)))
    g=grab(f"blk.{n}.ssm_a")
    q=-torch.exp(get(QB,f"{L}.{n}.linear_attn.A_log").float())
    o=-torch.exp(get(OX,f"{L}.{n}.linear_attn.A_log").float())
    print("  blk.%-2d %-21s %-14s %-14s" % (n,"ssm_a (vs -exp(A_log))",ms(g,q),ms(g,o)))
print("\n=== THE ONE TENSOR FAMILY THAT IS A REAL DIFFERENCE ===")
neq=0; tot=0
for n in [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30]:
    g=grab(f"blk.{n}.ssm_norm.weight")
    q=get(QB,f"{L}.{n}.linear_attn.norm.weight").float(); o=get(OX,f"{L}.{n}.linear_attn.norm.weight").float()
    tot+=1; neq+= torch.equal(g,o)
    if not torch.equal(g,o): print("   blk",n,"MISMATCH vs OxCoder")
print("  ssm_norm == local OxCoder : %d/%d" % (neq,tot))
print("  ssm_norm == local Qwen    : %d/%d" % (sum(torch.equal(grab(f'blk.{n}.ssm_norm.weight'),get(QB,f'{L}.{n}.linear_attn.norm.weight').float()) for n in [0,2,12]),3),"(sampled 3)")
