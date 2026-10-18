"""How big is DeltaCoder's LoRA delta compared to a KNOWN-GOOD fine-tune's?"""
import json, glob, torch, statistics
from safetensors import safe_open
AD="/mnt/sdc/oxopus/deltacoder"
cfg=json.load(open(AD+"/adapter_config.json")); scale=cfg["lora_alpha"]/cfg["r"]
ad={}
with safe_open(AD+"/adapter_model.safetensors","pt") as h:
    for k in h.keys(): ad[k]=h.get_tensor(k)
def load(d):
    t={}
    for f in sorted(glob.glob(d+"/*.safetensors")):
        with safe_open(f,"pt") as h:
            for k in h.keys(): t[k]=f
    return t
QB=load("/mnt/sdc/oxopus/models/Qwen3.5-9B"); QW=load("/mnt/sdc/oxopus/models/Qwopus3.5-9B-Coder")
hs={}
def get(idx,k):
    if idx[k] not in hs: hs[idx[k]]=safe_open(idx[k],"pt")
    return hs[idx[k]].get_tensor(k)
keys=sorted({k.split(".lora_")[0] for k in ad if ".lora_" in k})
def cand(k):
    b=k.replace("base_model.model.","")
    return "model.language_model."+b[len("model."):]+".weight"
lora=[]; qwopus=[]; both=[]
for key in keys:
    hk=cand(key)
    if hk not in QB or hk not in QW: continue
    A=ad.get(key+".lora_A.weight"); B=ad.get(key+".lora_B.weight")
    if A is None or B is None: continue
    W=get(QB,hk)
    if W.dtype!=torch.bfloat16: continue
    d=scale*(B.float()@A.float())
    if d.shape!=W.shape:
        if d.t().shape==W.shape: d=d.t()
        else: continue
    Wn=float(W.float().norm())
    dq=float((get(QW,hk).float()-W.float()).norm())
    lora.append(float(d.norm())/Wn); qwopus.append(dq/Wn)
    both.append((hk,float(d.norm())/Wn,dq/Wn))
def st(v): 
    v=sorted(v); return v[0],v[len(v)//2],v[-1]
lm=st(lora); qm=st(qwopus)
print("Relative delta ||d||/||W|| over the %d modules the adapter targets:" % len(lora))
print("  %-34s %10s %10s %10s" % ("","min","median","max"))
print("  %-34s %10.6f %10.6f %10.6f" % ("DeltaCoder LoRA (scale=%.0f)"%scale,*lm))
print("  %-34s %10.6f %10.6f %10.6f" % ("Qwopus fine-tune (known good)",*qm))
print("\n  ratio Qwopus/LoRA at the median: %.1fx" % (qm[1]/lm[1] if lm[1] else 0))
print("\n  per-module sample:")
print("  %-42s %11s %11s %8s" % ("module","LoRA","Qwopus","ratio"))
for hk,a,b in both[:10]:
    print("  %-42s %11.6f %11.6f %7.1fx" % (hk.replace("model.language_model.","")[-42:],a,b,(b/a if a else 0)))
big=[b for _,a,b in both if b>0]
print("\n  modules where Qwopus itself moved nothing: %d/%d" % (len(both)-len(big),len(both)))
