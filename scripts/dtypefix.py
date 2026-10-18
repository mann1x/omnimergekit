import glob, torch, collections
from safetensors import safe_open
R="/mnt/sdc/oxopus/models"
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
f32=[k for k in sorted(QB) if get(QB,k).dtype==torch.float32]
print("Qwen F32 tensors: %d" % len(f32))
fam=collections.Counter(k.split(".")[-2]+"."+k.split(".")[-1] for k in f32)
for a,b in fam.most_common(): print("   %-34s %d" % (a,b))
print()
print("=== is OxCoder simply Qwen downcast to bf16? ===")
exact=downcast=other=0; ex=[]
for k in sorted(OX):
    o=get(OX,k); q=get(QB,k)
    if o.dtype==q.dtype and torch.equal(o,q): exact+=1
    elif torch.equal(o, q.to(o.dtype)): downcast+=1
    else:
        other+=1; ex.append((k,float((o.float()-q.float()).abs().max())))
print("  bit-identical (same dtype)      : %d" % exact)
print("  == Qwen DOWNCAST to bf16        : %d" % downcast)
print("  genuinely different             : %d" % other)
for k,d in ex[:8]: print("     %-64s %.6f" % (k[:64],d))
print()
print("=== CORRECTED float32 comparison (the right way) ===")
same=0; diff=[]
for k in sorted(OX):
    o=get(OX,k).float(); q=get(QB,k).float()
    if torch.equal(o,q): same+=1
    else: diff.append((k,float((o-q).abs().max())))
print("  identical in f32 : %d / %d" % (same,len(OX)))
print("  differing in f32 : %d" % len(diff))
tot=sum(get(OX,k).numel() for k in OX); dp=sum(get(OX,k).numel() for k,_ in diff)
print("  differing params : %d / %d = %.6f%%" % (dp,tot,100*dp/tot))
