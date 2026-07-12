#!/usr/bin/env python3
"""Patch generate_cd_maps.py CD_TIERS: the sub-Q4 *i-quant* low slots ruminate on
the 98e-pruned Gemma-4 MoE (IQ3_S on ffn_gate_up_exps/gate/up destroys stop-token
emission — proven by termination smoke 2026-06-06: plain Q4_K_M 5/5 STOP, CD with
attention=Q5_K but FFN=IQ3_S 5/5 RUMINATE, CD with FFN=Q3_K 5/5 STOP). Swap the
fragile i-quant low slots for robust k-quants. The 'IQ codebook preserves quality'
rationale (lines 51-66) was HE+ pass@1 on a salad-extractor, NOT termination.

  CD-Q4_K_M  low IQ3_S -> Q3_K   (validated 5/5 STOP, 11 GB)
  CD-Q3_K_L  low IQ3_S -> Q3_K
  CD-Q2_K    low IQ2_S -> Q2_K
  CD-IQ4_K_M low IQ3_S -> Q3_K   (becomes IQ4_NL/IQ4_XS/Q3_K mixed)

Opt-in/known-broken tiers (CD-IQ3_K_M, CD-IQ2_K, lines 80-81) are left untouched.
Idempotent: refuses if the target lines are already swapped.
"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                 else "/srv/ml/repos/omnimergekit/scripts/generate_cd_maps.py")
src = p.read_text()
orig = src

swaps = [
    ('    "CD-Q4_K_M": ("Q5_K", "Q4_K", "IQ3_S"),',
     '    "CD-Q4_K_M": ("Q5_K", "Q4_K", "Q3_K"),  # was IQ3_S — ruminated on 98e MoE (smoke 2026-06-06)'),
    ('    "CD-Q3_K_L": ("Q4_K", "Q3_K", "IQ3_S"),',
     '    "CD-Q3_K_L": ("Q4_K", "Q3_K", "Q3_K"),  # was IQ3_S — ruminated on 98e MoE'),
    ('    "CD-Q2_K":   ("Q3_K", "Q2_K", "IQ2_S"),',
     '    "CD-Q2_K":   ("Q3_K", "Q2_K", "Q2_K"),  # was IQ2_S — i-quant low ruminates'),
    ('    "CD-IQ4_K_M": ("IQ4_NL", "IQ4_XS", "IQ3_S"),   # ~10-11 GB target band',
     '    "CD-IQ4_K_M": ("IQ4_NL", "IQ4_XS", "Q3_K"),   # ~10-11 GB; low IQ3_S->Q3_K (ruminated)'),
]

applied = 0
for old, new in swaps:
    if old in src:
        src = src.replace(old, new)
        applied += 1
    else:
        print(f"  [skip] not found (already patched?): {old.strip()[:50]}")

if src == orig:
    sys.exit("FATAL: no CD_TIERS low-slot swap applied (already patched?)")
pathlib.Path(str(p) + ".bak_lowtier").write_text(orig)
p.write_text(src)
print(f"PATCHED {p}: {applied}/4 low-slot i-quant -> k-quant swaps")
