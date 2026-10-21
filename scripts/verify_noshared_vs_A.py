"""Prove A-noshared is map A minus the shared upweight, bit for bit.

Every tensor of gemma-4-A4B-20B-v9-agentic-A-noshared must equal map A's, except the keys ending in
mlp.down_proj.weight, where A must equal bf16(noshared.float() * 1.2) exactly (router_shared_upweight.py L122).
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

W = Path(sys.argv[1])
a_dir, n_dir = W / "gemma-4-A4B-20B-v9-agentic-A", W / "gemma-4-A4B-20B-v9-agentic-A-noshared"
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
    if k.endswith("mlp.down_proj.weight"):
        ok = torch.equal(ta, (tn.to(torch.float32) * 1.2).to(tn.dtype))
        scaled += ok
        differs = not torch.equal(ta, tn)
        if not ok or not differs:
            bad.append((k, "scaled-rule" if not ok else "unchanged"))
    else:
        ok = torch.equal(ta, tn)
        same += ok
        if not ok:
            bad.append((k, "differs"))
print(f"tensors {len(ia)}: identical {same}, shared down_proj == bf16(x*1.2) {scaled}, bad {len(bad)}")
for b in bad[:10]:
    print("  BAD", b)
print("VERIFY_PASS" if not bad and scaled == 30 else "VERIFY_FAIL")
