import json, glob, torch, sys
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
BASES={"Qwen3.5-9B":load("/mnt/sdc/oxopus/models/Qwen3.5-9B"),
       "Qwopus3.5-9B-Coder":load("/mnt/sdc/oxopus/models/Qwopus3.5-9B-Coder")}
hs={}
def get(idx,k):
    if idx[k] not in hs: hs[idx[k]]=safe_open(idx[k],"pt")
    return hs[idx[k]].get_tensor(k)
keys=sorted({k.split(".lora_")[0] for k in ad if ".lora_" in k})
print("raw adapter key sample:", keys[0])
ref=BASES["Qwen3.5-9B"]
def cands(k):
    b=k.replace("base_model.model.","")
    out=[b+".weight"]
    if b.startswith("model.layers."):
        out.insert(0,"model.language_model."+b[len("model."):]+".weight")
    if b.startswith("layers."):
        out.insert(0,"model.language_model."+b+".weight")
    return out
print("candidates:", cands(keys[0]))
matched=[k for k in keys if any(c in ref for c in cands(k))]
print("MATCHED %d/%d adapter targets against the base" % (len(matched),len(keys)))
if not matched:
    print("REFUSE: zero matches -- mapping is broken, no measurement made"); sys.exit(1)
for name,idx in BASES.items():
    tot_e=tot_ch=0; rels=[]; shown=0; nmod=0
    print(f"\n=== merge {len(matched)} LoRA deltas into {name} (bf16) ===")
    print("  %-44s %11s %9s %s" % ("module","||d||/||W||","med d/ULP","entries changed"))
    for key in matched:
        A=ad.get(key+".lora_A.weight"); B=ad.get(key+".lora_B.weight")
        if A is None or B is None: continue
        hk=next(c for c in cands(key) if c in idx)
        W=get(idx,hk)
        if W.dtype!=torch.bfloat16: continue
        d=scale*(B.float()@A.float())
        if d.shape!=W.shape:
            if d.t().shape==W.shape: d=d.t()
            else: continue
        Wf=W.float(); merged=(Wf+d).to(torch.bfloat16)
        ch=int((merged!=W).sum()); n=W.numel()
        tot_e+=n; tot_ch+=ch; nmod+=1
        rel=float(d.norm()/Wf.norm()); rels.append(rel)
        ulp=(Wf.abs()*(2.0**-8)).clamp(min=1e-30)
        if shown<8:
            print("  %-44s %11.6f %9.3f %8d/%d (%.2f%%)" %
                  (hk.replace("model.language_model.","")[-44:],rel,
                   float((d.abs()/ulp).median()),ch,n,100*ch/n)); shown+=1
    rels.sort()
    print("  modules merged: %d   ||d||/||W|| min/med/max = %.6f / %.6f / %.6f" %
          (nmod, rels[0], rels[len(rels)//2], rels[-1]))
    print("  TOTAL entries changed: %d / %d = %.4f%%" % (tot_ch,tot_e,100*tot_ch/tot_e))
    print("  VERDICT:", "NO-OP (delta below bf16 resolution)" if 100*tot_ch/tot_e<1 else "DELTA SURVIVES")
