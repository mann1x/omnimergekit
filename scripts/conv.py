"""Are the 'differing' tensors conversion conventions, or real weight changes?"""
import json, glob, urllib.request, numpy as np, torch
from safetensors import safe_open
H=json.load(open("/mnt/sdc/oxopus/gguf_header.json")); URL,DS=H["url"],H["data_start"]
INFO={i["name"]:i for i in H["infos"]}; R="/mnt/sdc/oxopus/models"
def grab(n,nb=None):
    i=INFO[n]; esz=4 if i["type"]==0 else 2
    N=int(np.prod(i["dims"]))*esz; N=N if nb is None else min(nb,N); N-=N%esz
    s=DS+i["offset"]
    r=urllib.request.Request(URL,headers={"Range":f"bytes={s}-{s+N-1}"})
    raw=urllib.request.urlopen(r,timeout=120).read()
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
print("=== hypothesis 1: GGUF norm = HF norm + 1  (Gemma/Qwen3.5 RMSNorm convention) ===")
for g,h in [("blk.0.attn_norm.weight",f"{L}.0.input_layernorm.weight"),
            ("blk.7.attn_q_norm.weight",f"{L}.7.self_attn.q_norm.weight"),
            ("output_norm.weight","model.language_model.norm.weight")]:
    a=grab(g); q=get(QB,h).float(); o=get(OX,h).float()
    print("  %-26s  g==q+1: %-5s   g==o+1: %-5s   max|g-(q+1)|=%.3e" %
          (g, torch.equal(a,q+1), torch.equal(a,o+1), (a-(q+1)).abs().max()))
print("\n=== hypothesis 2: ssm_a = -exp(A_log) ===")
for n in [0,2,12]:
    a=grab(f"blk.{n}.ssm_a"); q=get(QB,f"{L}.{n}.linear_attn.A_log").float(); o=get(OX,f"{L}.{n}.linear_attn.A_log").float()
    print("  blk.%-2d  g==-exp(q): %-5s  g==-exp(o): %-5s  max|g-(-exp(q))|=%.3e"
          % (n, torch.equal(a,-torch.exp(q)), torch.equal(a,-torch.exp(o)), (a-(-torch.exp(q))).abs().max()))
print("\n=== hypothesis 3: ssm_conv1d is a LAYOUT change (HF [C,1,K] -> GGUF [K,C]) ===")
for n in [0,2]:
    a=grab(f"blk.{n}.ssm_conv1d.weight"); q=get(QB,f"{L}.{n}.linear_attn.conv1d.weight").float()
    o=get(OX,f"{L}.{n}.linear_attn.conv1d.weight").float()
    qt=q.squeeze(1).t().contiguous().flatten(); ot=o.squeeze(1).t().contiguous().flatten()
    print("  blk.%-2d  HF%s -> flat==GGUF: %-5s   (vs OxCoder: %s)" % (n,list(q.shape),torch.equal(a,qt),torch.equal(a,ot)))
print("\n=== hypothesis 4: ssm_dt.bias / ssm_out are plain copies? ===")
for n in [0,2]:
    a=grab(f"blk.{n}.ssm_dt.bias"); q=get(QB,f"{L}.{n}.linear_attn.dt_bias").float()
    print("  blk.%-2d dt_bias  g==q:%-5s  g==softplus-inv?  max|g-q|=%.4f  g[:3]=%s q[:3]=%s"
          % (n,torch.equal(a,q),(a-q).abs().max(),[round(x,3) for x in a[:3].tolist()],[round(x,3) for x in q[:3].tolist()]))
    b=grab(f"blk.{n}.ssm_out.weight",64*1024); qo=get(QB,f"{L}.{n}.linear_attn.out_proj.weight")
    print("         ssm_out  g==q_prefix:%-5s  g==q_T_prefix:%-5s" %
          (torch.equal(b,qo.flatten()[:b.numel()]), torch.equal(b,qo.t().contiguous().flatten()[:b.numel()])))
