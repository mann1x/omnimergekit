import json, glob, os
from safetensors import safe_open
ROOT = "/mnt/sdc/oxopus/models"
for m in ["Qwopus3.5-9B-Coder", "OxCoder-9B", "Qwen3.5-9B"]:
    d = os.path.join(ROOT, m)
    files = sorted(glob.glob(d + "/*.safetensors"))
    names, bad = [], []
    for f in files:
        try:
            with safe_open(f, "pt") as h:
                names += list(h.keys())
        except Exception as e:
            bad.append(os.path.basename(f) + ": " + str(e))
    mtp = [n for n in names if n.startswith("mtp.")]
    vis = [n for n in names if n.startswith("visual.")]
    idx = os.path.join(d, "model.safetensors.index.json")
    nidx = len(json.load(open(idx))["weight_map"]) if os.path.exists(idx) else "no-index"
    cfg = json.load(open(os.path.join(d, "config.json")))
    eos = cfg.get("eos_token_id")
    print("%-22s files=%d tensors=%4d idx=%s mtp=%3d visual=%3d eos=%s vocab=%s" %
          (m, len(files), len(names), nidx, len(mtp), len(vis), eos, cfg.get("vocab_size")))
    if bad:
        print("   *** CORRUPT ***", bad)
