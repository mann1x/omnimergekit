import json, glob, os, hashlib, torch
from safetensors import safe_open
ROOT="/mnt/sdc/oxopus/models"; OUT="/mnt/sdc/oxopus/OxOpusCoder-9B"
def load(d):
    t={}
    for f in sorted(glob.glob(d+"/*.safetensors")):
        with safe_open(f,"pt") as h:
            for k in h.keys(): t[k]=f
    return t
out=load(OUT); base=load(ROOT+"/Qwopus3.5-9B-Coder")
print("tensors: out=%d base=%d" % (len(out),len(base)))
for pfx in ("mtp.","visual."):
    ko=[k for k in out if pfx in k]; print("  %-9s out=%3d base=%3d" % (pfx,len(ko),len([k for k in base if pfx in k])))
def get(t,k):
    with safe_open(t[k],"pt") as h: return h.get_tensor(k)
bad=[]; checked=0
for k in sorted(out):
    if "mtp." in k or "visual." in k:
        a,b=get(out,k),get(base,k); checked+=1
        if not torch.equal(a,b): bad.append(k)
print("  bit-identical to base: %d/%d protected tensors  %s" % (checked-len(bad),checked,"OK" if not bad else "*** DRIFT: "+str(bad[:3])))
# a merged tensor MUST differ, else the merge was a no-op
mk=[k for k in sorted(out) if "mtp." not in k and "visual." not in k and k.endswith("mlp.up_proj.weight")][:3]
for k in mk:
    d=(get(out,k).float()-get(base,k).float()).abs().max().item()
    print("  merged-delta %-58s max|out-base|=%.6f %s" % (k[:58],d,"OK" if d>0 else "*** NO-OP ***"))
cfg=json.load(open(OUT+"/config.json")); print("  root eos_token_id =",cfg.get("eos_token_id"),"(need 248046)")
for f in ("tokenizer.json","tokenizer_config.json","chat_template.jinja"):
    def sh(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:16] if os.path.exists(p) else "absent"
    print("  %-22s out=%s qwopus=%s oxcoder=%s" % (f,sh(OUT+"/"+f),sh(ROOT+"/Qwopus3.5-9B-Coder/"+f),sh(ROOT+"/OxCoder-9B/"+f)))
