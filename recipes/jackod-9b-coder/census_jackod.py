import sys, glob, os
from safetensors import safe_open
def names(d):
    s = set()
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            s |= set(h.keys())
    return s
b, o = names(sys.argv[1]), names(sys.argv[2])
mtp = sum(1 for k in o if k.startswith("mtp.") or ".mtp." in k)
print("  base=%d out=%d equal=%s  mtp=%d" % (len(b), len(o), b == o, mtp))
# a gate, not a report: the output must carry exactly the base's tensors, MTP head (15) included
sys.exit(0 if b == o and mtp == 15 else 1)
