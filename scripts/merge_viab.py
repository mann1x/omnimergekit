"""Does the DeltaCoder LoRA delta SURVIVE being merged into bf16 weights?

The OxCoder lesson: a delta below half a bf16 ULP is not a small contribution,
it is exactly zero. So measure the merge, don't estimate it.
"""
import json, glob, torch
from safetensors import safe_open

AD = "/mnt/sdc/oxopus/deltacoder"
cfg = json.load(open(AD + "/adapter_config.json"))
scale = cfg["lora_alpha"] / cfg["r"]
print(f"r={cfg['r']} alpha={cfg['lora_alpha']} scale={scale}  rslora={cfg.get('use_rslora')}")

ad = {}
with safe_open(AD + "/adapter_model.safetensors", "pt") as h:
    for k in h.keys(): ad[k] = h.get_tensor(k)

def load(d):
    t = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys(): t[k] = f
    return t
BASES = {"Qwen3.5-9B": load("/mnt/sdc/oxopus/models/Qwen3.5-9B"),
         "Qwopus3.5-9B-Coder": load("/mnt/sdc/oxopus/models/Qwopus3.5-9B-Coder")}
hs = {}
def get(idx, k):
    if idx[k] not in hs: hs[idx[k]] = safe_open(idx[k], "pt")
    return hs[idx[k]].get_tensor(k)

keys = sorted({k.split(".lora_")[0] for k in ad if ".lora_" in k})
print(f"adapter targets: {len(keys)}\n")

def hfname(k):
    k = k.replace("base_model.model.", "")
    return "model.language_model." + k if k.startswith("layers.") else k

rows = []
for name, idx in BASES.items():
    tot_e = tot_ch = 0
    print(f"=== merging into {name} (bf16) ===")
    print("  %-46s %10s %10s %12s" % ("module", "||d||/||W||", "d/ULP", "entries changed"))
    shown = 0
    for key in keys:
        A = ad.get(key + ".lora_A.weight"); B = ad.get(key + ".lora_B.weight")
        if A is None or B is None: continue
        hk = hfname(key) + ".weight"
        if hk not in idx: continue
        W = get(idx, hk)
        if W.dtype != torch.bfloat16: continue
        d = (scale * (B.float() @ A.float()))
        if d.shape != W.shape:
            if d.t().shape == W.shape: d = d.t()
            else: continue
        Wf = W.float()
        merged = (Wf + d).to(torch.bfloat16)
        ch = int((merged != W).sum()); n = W.numel()
        tot_e += n; tot_ch += ch
        rel = float(d.norm() / Wf.norm())
        # bf16 ULP at each weight's magnitude
        ulp = (Wf.abs() * (2.0 ** -8)).clamp(min=1e-30)
        ratio = float((d.abs() / ulp).median())
        if shown < 6:
            print("  %-46s %10.6f %10.3f %8d/%d (%.2f%%)" %
                  (hfname(key)[-46:], rel, ratio, ch, n, 100 * ch / n))
            shown += 1
    print("  %-46s %10s %10s %8d/%d (%.4f%%)\n" %
          ("TOTAL over adapter-targeted weights", "", "", tot_ch, tot_e,
           100 * tot_ch / tot_e if tot_e else 0))
    rows.append((name, tot_ch, tot_e))
print("VERDICT")
for name, ch, e in rows:
    p = 100 * ch / e if e else 0
    print(f"  {name:<24} {ch:,}/{e:,} entries change = {p:.4f}%  "
          + ("-> MERGE IS A NO-OP" if p < 1.0 else "-> delta survives"))
