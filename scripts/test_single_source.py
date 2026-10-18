"""Does omnimerge_v2 degenerate CORRECTLY at K=1, or silently distort?

Claim under test: with one source the EMR election is an identity, so the
result must equal  base + w * DAREx-rescaled-topk(source - task_base).
If that equality does not hold to bit-level, the single-source path is NOT
safe and the gate should stay.
"""
import sys, torch
sys.path.insert(0, "/srv/ml/repos/omnimergekit")
from omnimergekit import _omnimerge_v2_chunk

torch.manual_seed(0)
N, density, q, w = 4096, 0.53, 0.75, 0.40
base = torch.randn(N, dtype=torch.float32)
tbase = torch.randn(N, dtype=torch.float32)
src = tbase + torch.randn(N) * 0.1
g = torch.Generator().manual_seed(42)

got = _omnimerge_v2_chunk(base, [src], [w], density, q, generator=g,
                          task_base_chunk=tbase)

# independent reference implementation of the claim
delta = src - tbase
k = max(1, int(density * N))
_, idx = torch.topk(delta.abs(), k, largest=True, sorted=False)
mask = torch.zeros(N, dtype=torch.bool); mask[idx] = True
ref = base + w * torch.where(mask, delta / q, torch.zeros_like(delta))

diff = (got - ref).abs().max().item()
print("max |omnimerge_v2(K=1) - reference| =", diff)
print("nonzero deltas applied:", int(mask.sum()), "of", N, "(density", density, ")")
print("VERDICT:", "IDENTITY - single source is mathematically safe" if diff == 0.0
      else "*** DISTORTS - do NOT relax the gate ***")
