#!/usr/bin/env python3
"""Prove omnimerge_v2's EMR election degenerates to an IDENTITY at one source.

This is the evidence behind relaxing the blanket `len(--source) < 2` refusal in
omnimergekit.py to a method-aware one. A 2-model merge (one --base, one
--source) is a legitimate request; the old gate refused it for EVERY method
while printing a TIES-specific message, so no two-model merge was possible at
all.

The claim: with a single source, omnimerge_v2 must reduce EXACTLY to

    base + w * where(topk_mask(|delta|, density), delta / darex_q, 0)

because EMR takes its sign from sum(stack, dim=0) and its amplitude from the
max-abs over dim 0, and both are the single delta when the stack has depth 1.
Anything short of bit-exact equality means the single-source path distorts the
merge and the gate must go back.
"""
import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from omnimergekit import _omnimerge_v2_chunk  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    N, density, q, w = 4096, 0.53, 0.75, 0.40
    base = torch.randn(N, dtype=torch.float32)
    tbase = torch.randn(N, dtype=torch.float32)
    src = tbase + torch.randn(N) * 0.1

    got = _omnimerge_v2_chunk(
        base, [src], [w], density, q,
        generator=torch.Generator().manual_seed(42), task_base_chunk=tbase)

    delta = src - tbase
    k = max(1, int(density * N))
    _, idx = torch.topk(delta.abs(), k, largest=True, sorted=False)
    mask = torch.zeros(N, dtype=torch.bool)
    mask[idx] = True
    ref = base + w * torch.where(mask, delta / q, torch.zeros_like(delta))

    diff = (got - ref).abs().max().item()
    print(f"  max |omnimerge_v2(K=1) - reference| = {diff}")
    print(f"  deltas applied: {int(mask.sum())}/{N} (density {density})")
    if diff != 0.0:
        print("FAIL: single-source omnimerge_v2 DISTORTS -- restore the >=2 gate")
        return 1
    print("PASS: EMR is an identity at one source; single-source merge is safe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
