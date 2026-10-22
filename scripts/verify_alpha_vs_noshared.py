"""Prove an alpha variant is A-noshared plus the shared upweight at ALPHA, bit for bit.

Usage: verify_alpha_vs_noshared.py <v9_agentic_ab dir> <variant dir name> <alpha> [layers]
Every tensor must equal A-noshared's, except the keys ending in mlp.down_proj.weight of the upweighted layers
(all 30, or the router_shared_upweight --layers spec given, e.g. 0-9), where the variant must equal
bf16(noshared.float() * alpha) exactly (router_shared_upweight.py).
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

W = Path(sys.argv[1])
a_dir, n_dir = W / sys.argv[2], W / "gemma-4-A4B-20B-v9-agentic-A-noshared"
ALPHA = float(sys.argv[3])
LAYERS = set(range(30))
if len(sys.argv) > 4:
    LAYERS = set()
    for part in sys.argv[4].split(","):
        lo, _, hi = part.partition("-")
        LAYERS.update(range(int(lo), int(hi or lo) + 1))
ia = json.loads((a_dir / "model.safetensors.index.json").read_text())["weight_map"]
ib = json.loads((n_dir / "model.safetensors.index.json").read_text())["weight_map"]
assert set(ia) == set(ib), f"key sets differ: {len(set(ia) ^ set(ib))}"
same = scaled = 0
bad = []
handles = {}


def get(d, idx, k):
    f = d / idx[k]
    if f not in handles:
        handles[f] = safe_open(str(f), "pt")
    return handles[f].get_tensor(k)


for k in sorted(ia):
    ta, tn = get(a_dir, ia, k), get(n_dir, ib, k)
    m = re.search(r"\.layers\.(\d+)\.", k)
    if k.endswith("mlp.down_proj.weight") and m and int(m.group(1)) in LAYERS:
        ok = torch.equal(ta, (tn.to(torch.float32) * ALPHA).to(tn.dtype))
        scaled += ok
        differs = not torch.equal(ta, tn)
        if not ok or not differs:
            bad.append((k, "scaled-rule" if not ok else "unchanged"))
    else:
        ok = torch.equal(ta, tn)
        same += ok
        if not ok:
            bad.append((k, "differs"))
print(f"tensors {len(ia)}: identical {same}, shared down_proj == bf16(x*alpha) {scaled} "
      f"(layers {sorted(LAYERS)}), bad {len(bad)}")
for b in bad[:10]:
    print("  BAD", b)
print("VERIFY_PASS" if not bad and scaled == len(LAYERS) else "VERIFY_FAIL")
