"""verify_expert_drop.py — prove a pruned Gemma-4 A4B build against its 128-expert base, tensor by tensor.

For every layer, from the build's own expert_drop_metadata.json (per_layer_keep / per_layer_drop):
  * keep ∪ drop = 0..E-1, disjoint, keep sorted, |keep| == config num_experts;
  * experts.gate_up_proj, experts.down_proj, router.proj.weight, router.per_expert_scale
    == base[keep] bit-exactly (row j of the router and expert j are the same original expert);
every other tensor (attention, norms, router.scale, layer_scalar, embeddings, vision, ...) is bit-identical to
the base, except mlp.down_proj.weight (the shared expert), reported as its ratio to the base (expected alpha);
no tensor keeps an E-sized axis; the config differs from the base only where listed.
usage: verify_expert_drop.py <base_dir> <pruned_dir> [--alpha 1.2]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

PER_EXPERT = re.compile(r"model\.language_model\.layers\.(\d+)\.(experts\.gate_up_proj|experts\.down_proj|"
                        r"router\.proj\.weight|router\.per_expert_scale)$")
SHARED_DOWN = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight$")


class Store:
    def __init__(self, d: Path):
        self.d = d
        self.wm = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
        self.h = {}

    def get(self, k):
        s = self.wm[k]
        if s not in self.h:
            self.h[s] = safe_open(str(self.d / s), framework="pt", device="cpu")
        return self.h[s].get_tensor(k)


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("pruned")
    ap.add_argument("--alpha", type=float, default=1.2)
    a = ap.parse_args()
    base, pr = Store(Path(a.base)), Store(Path(a.pruned))
    meta = json.load(open(Path(a.pruned) / "expert_drop_metadata.json"))
    cb, cp = (flat(json.load(open(Path(x) / "config.json"))) for x in (a.base, a.pruned))
    E, N = cb["text_config.num_experts"], cp["text_config.num_experts"]
    bad = []
    cfg_diff = {k: (cb.get(k), cp.get(k)) for k in sorted(set(cb) | set(cp)) if cb.get(k) != cp.get(k)}
    print(f"experts {E} -> {N}; config keys that differ from the base: {cfg_diff}")
    keep = {int(k): v for k, v in meta["per_layer_keep"].items()}
    drop = {int(k): v for k, v in meta["per_layer_drop"].items()}
    for li in keep:
        if keep[li] != sorted(keep[li]) or len(keep[li]) != N or set(keep[li]) & set(drop[li]) \
                or set(keep[li]) | set(drop[li]) != set(range(E)):
            bad.append(f"L{li}: keep/drop index lists inconsistent")
    if set(base.wm) != set(pr.wm):
        bad.append(f"key sets differ: only base {sorted(set(base.wm) - set(pr.wm))[:5]}, "
                   f"only pruned {sorted(set(pr.wm) - set(base.wm))[:5]}")
    n_pe = n_same = 0
    ratios = []
    for k in sorted(set(base.wm) & set(pr.wm)):
        tb, tp = base.get(k), pr.get(k)
        m = PER_EXPERT.match(k)
        if m:
            idx = torch.tensor(keep[int(m.group(1))])
            if tp.shape[0] != N or not torch.equal(tp, tb[idx]):
                bad.append(f"{k}: != base[keep] (shape {list(tp.shape)})")
            n_pe += 1
            continue
        if E in tp.shape and E not in (2816, 704, 1408):
            bad.append(f"{k}: still has an {E}-sized axis {list(tp.shape)}")
        if SHARED_DOWN.match(k):
            ref = (tb.float() * a.alpha).to(tb.dtype)
            r = (tp.float().norm() / tb.float().norm()).item()
            ratios.append(r)
            if not torch.equal(tp, ref):
                d = (tp.float() - ref.float()).abs().max().item()
                bad.append(f"{k}: != bf16(base x {a.alpha}) (max |diff| {d:.3g}, norm ratio {r:.4f})")
            continue
        if tp.shape != tb.shape or not torch.equal(tp, tb):
            bad.append(f"{k}: differs from the base (shape {list(tb.shape)} -> {list(tp.shape)})")
        n_same += 1
    print(f"per-expert tensors checked {n_pe} (expect {4 * len(keep)}); identical-to-base tensors {n_same}; "
          f"shared down_proj norm ratio min {min(ratios):.4f} max {max(ratios):.4f} over {len(ratios)} layers")
    for b in bad:
        print("BAD", b)
    print("VERIFY_OK" if not bad and n_pe == 4 * len(keep) else f"VERIFY_FAIL ({len(bad)} problems)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
