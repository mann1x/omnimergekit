"""Census a REMOTE safetensors model against a local reference, via HTTP Range.
Safetensors: u64 header-len, JSON header {name:{dtype,shape,data_offsets}}, then data."""
import json, struct, sys, glob, urllib.request, numpy as np, torch, collections
from safetensors import safe_open

REPO = sys.argv[1] if len(sys.argv) > 1 else "Tesslate/OmniCoder-9B"
FILE = sys.argv[2] if len(sys.argv) > 2 else "model.safetensors"
REF  = sys.argv[3] if len(sys.argv) > 3 else "/mnt/sdc/oxopus/models/Qwen3.5-9B"
URL = f"https://huggingface.co/{REPO}/resolve/main/{FILE}"
FETCHED = [0]

def rng(start, n):
    r = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start+n-1}"})
    for _ in range(4):
        try:
            with urllib.request.urlopen(r, timeout=180) as h: b = h.read()
            FETCHED[0] += len(b); return b
        except Exception as e: err = e
    raise SystemExit(f"range fail {start}+{n}: {err}")

n_hdr = struct.unpack("<Q", rng(0, 8))[0]
hdr = json.loads(rng(8, n_hdr))
DATA = 8 + n_hdr
meta = hdr.pop("__metadata__", None)
print(f"remote : {URL}")
print(f"  header {n_hdr/1e6:.2f} MB, {len(hdr)} tensors, __metadata__={meta}")
DT = {"BF16": (torch.bfloat16, 2), "F16": (torch.float16, 2), "F32": (torch.float32, 4)}
dts = collections.Counter(v["dtype"] for v in hdr.values())
print(f"  dtypes {dict(dts)}")

def fetch(name, maxb=None):
    e = hdr[name]; td, esz = DT[e["dtype"]]
    s, t = e["data_offsets"]; n = t - s
    if maxb: n = min(n, maxb)
    n -= n % esz
    raw = rng(DATA + s, n)
    a = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
    return a.view(td)

ref = {}
for f in sorted(glob.glob(REF + "/*.safetensors")):
    with safe_open(f, "pt") as h:
        for k in h.keys(): ref[k] = f
hs = {}
def rget(k):
    if ref[k] not in hs: hs[ref[k]] = safe_open(ref[k], "pt")
    return hs[ref[k]].get_tensor(k)

common = [k for k in hdr if k in ref]
only_r = [k for k in hdr if k not in ref]
print(f"  common with {REF.split('/')[-1]}: {len(common)}   only-in-remote: {len(only_r)}")
if only_r: print("   e.g.", only_r[:5])

NB = 64 * 1024
same = 0; diff = []; grp = collections.Counter()
for k in sorted(common):
    g = fetch(k, NB)
    r = rget(k).flatten()[:g.numel()]
    if r.dtype != g.dtype: r = r.to(g.dtype)
    if torch.equal(g, r): same += 1
    else:
        d = float((g.float() - r.float()).abs().max())
        diff.append((k, d))
        p = k.split(".")
        grp["".join(["...", p[4]]) if len(p) > 4 else k] += 1
print(f"\nCENSUS (64 KB prefix per tensor) vs {REF.split('/')[-1]}")
print(f"  BIT-IDENTICAL : {same} / {len(common)}")
print(f"  DIFFERENT     : {len(diff)} / {len(common)}")
if diff:
    ds = [d for _, d in diff]
    print(f"  max|delta| range: {min(ds):.6f} .. {max(ds):.6f}   median {sorted(ds)[len(ds)//2]:.6f}")
    print("  differing tensor families:")
    for p, c in grp.most_common(14): print(f"    {p:<28} {c}")
    print("  sample:")
    for k, d in diff[:6]: print(f"    {k[:66]:<66} {d:.6f}")
if same and diff:
    print("\n  tensors that are IDENTICAL to the base (sample):")
    ident = [k for k in sorted(common) if k not in dict(diff)][:6]
    for k in ident: print("   ", k)
print(f"\nbytes fetched: {FETCHED[0]/1e6:.1f} MB  (file is 18.82 GB)")
