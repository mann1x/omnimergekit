import json,struct,urllib.request,numpy as np,torch,glob
from safetensors import safe_open
REPO="OrionLLM/OxCoder-9B"; FILE="model.safetensors"
URL=f"https://huggingface.co/{REPO}/resolve/main/{FILE}"
def rng(s,n):
    r=urllib.request.Request(URL,headers={"Range":f"bytes={s}-{s+n-1}"})
    with urllib.request.urlopen(r,timeout=180) as h: return h.read()
n_hdr=struct.unpack("<Q",rng(0,8))[0]
hdr=json.loads(rng(8,n_hdr)); hdr.pop("__metadata__",None)
DATA=8+n_hdr
K="model.language_model.layers.10.linear_attn.norm.weight"
e=hdr[K]; print("remote entry:",K); print("  ",e)
s,t=e["data_offsets"]
raw=rng(DATA+s,t-s)
g=torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy()).view(torch.bfloat16)
def loc(d):
    for f in sorted(glob.glob(d+"/*.safetensors")):
        with safe_open(f,"pt") as h:
            if K in h.keys(): return h.get_tensor(K)
lo=loc("/mnt/sdc/oxopus/models/OxCoder-9B"); lq=loc("/mnt/sdc/oxopus/models/Qwen3.5-9B")
print("  remote[:6] ",[round(x,6) for x in g[:6].float().tolist()])
print("  localOx[:6]",[round(x,6) for x in lo[:6].float().tolist()])
print("  localQw[:6]",[round(x,6) for x in lq[:6].float().tolist()])
print("  remote==localOxCoder:",torch.equal(g,lo))
print("  remote==localQwen   :",torch.equal(g,lq))
print("  localOx==localQwen  :",torch.equal(lo,lq))
print()
print("=== is the local Qwen dir REALLY qwen? spot-check a few ssm_norms ===")
for n in (10,12,2):
    kk=f"model.language_model.layers.{n}.linear_attn.norm.weight"
    a=loc("/mnt/sdc/oxopus/models/OxCoder-9B"); 
    with safe_open([f for f in sorted(glob.glob('/mnt/sdc/oxopus/models/Qwen3.5-9B/*.safetensors')) if kk in safe_open(f,'pt').keys()][0],"pt") as h:
        b=h.get_tensor(kk)
    with safe_open([f for f in sorted(glob.glob('/mnt/sdc/oxopus/models/OxCoder-9B/*.safetensors')) if kk in safe_open(f,'pt').keys()][0],"pt") as h:
        c=h.get_tensor(kk)
    print(f"  layer {n}: localOx==localQwen -> {torch.equal(c,b)}   max|d|={float((c.float()-b.float()).abs().max()):.6f}")
