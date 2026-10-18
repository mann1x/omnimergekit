import json, glob, os, torch
from safetensors import safe_open
R="/mnt/sdc/oxopus/models"
print("=== context / rope ===")
for m in ["Qwopus3.5-9B-Coder","OxCoder-9B","Qwen3.5-9B"]:
    c=json.load(open(f"{R}/{m}/config.json"))
    t=c.get("text_config",c)
    print("  %-22s max_pos=%-8s rope_theta=%-12s scaling=%s" % (
        m, t.get("max_position_embeddings"), t.get("rope_theta"), t.get("rope_scaling")))
print()
print("=== is OxCoder actually different from Qwen3.5-9B? ===")
def load(d):
    t={}
    for f in sorted(glob.glob(d+"/*.safetensors")):
        with safe_open(f,"pt") as h:
            for k in h.keys(): t[k]=f
    return t
ox=load(R+"/OxCoder-9B"); qb=load(R+"/Qwen3.5-9B"); qw=load(R+"/Qwopus3.5-9B-Coder")
def get(t,k):
    with safe_open(t[k],"pt") as h: return h.get_tensor(k)
common=[k for k in sorted(ox) if k in qb]
print("  common tensors OxCoder<->Qwen3.5-9B:", len(common))
import random; random.seed(0)
samp=[k for k in common if any(s in k for s in ("mlp.up_proj","self_attn.q_proj","mlp.down_proj","input_layernorm"))][:8]
nz=0
for k in samp:
    d=(get(ox,k).float()-get(qb,k).float()).abs().max().item()
    d2=(get(qw,k).float()-get(qb,k).float()).abs().max().item()
    if d>0: nz+=1
    print("    %-62s |Ox-Qwen|=%.6f  |Qwopus-Qwen|=%.6f" % (k[:62],d,d2))
