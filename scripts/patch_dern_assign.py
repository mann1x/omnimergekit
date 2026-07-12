#!/usr/bin/env python3
"""patch_dern_assign.py <redist_dern_eq11.py> — add soft top-k assignment knob.

Adds:
  --assign-topk INT  (default 0 = hard cosine-nearest = current baseline behavior)
                     >1 = each dropped expert is split across its top-k nearest survivors,
                     weighted by softmax(cosine); member weight = freq**exp * assignment_frac.

Idempotent + asserts each replacement is unique. Default 0 reproduces the T203/baseline fold
byte-for-byte at the algorithm level (hard argmax, each dropped -> exactly one survivor, frac 1.0).
"""
import ast
import sys

path = sys.argv[1]
src = open(path).read()
orig_len = len(src)

if "_assign_dropped_soft" in src and "--assign-topk" in src:
    print("already patched")
    sys.exit(0)


def repl(old, new, s):
    n = s.count(old)
    assert n == 1, f"expected 1 occurrence, found {n} of:\n---\n{old[:140]}\n---"
    return s.replace(old, new)


# 1. insert soft-assignment helper + extend fold_layer signature with assign_topk
src = repl(
    'def fold_layer(x, gu_t, dn_t, keep_ids, drop_ids, freq_l, device, stats, freq_exp=1.0, norm_anchor="survivor"):',
    'def _assign_dropped_soft(mean_out, keep_ids, drop_ids, topk=2):\n'
    '    """Soft top-k assignment: each dropped -> top-k survivors, frac = softmax(cosine).\n'
    '    Returns {survivor_id: [(dropped_id, frac), ...]}."""\n'
    '    mo = torch.nn.functional.normalize(mean_out.float(), dim=-1)\n'
    '    surv = torch.tensor(keep_ids)\n'
    '    assign = {i: [] for i in keep_ids}\n'
    '    k = min(topk, len(keep_ids))\n'
    '    for j in drop_ids:\n'
    '        sims = mo[surv] @ mo[j]\n'
    '        vals, idx = torch.topk(sims, k)\n'
    '        fr = torch.softmax(vals, dim=0)\n'
    '        for t in range(k):\n'
    '            assign[keep_ids[int(idx[t])]].append((j, float(fr[t])))\n'
    '    return assign\n'
    '\n'
    '\n'
    'def fold_layer(x, gu_t, dn_t, keep_ids, drop_ids, freq_l, device, stats, freq_exp=1.0, norm_anchor="survivor", assign_topk=0):',
    src)

# 2. assignment branch (hard -> uniform (id,1.0) pairs; soft -> (id,frac) pairs)
src = repl(
    "    assign = _assign_dropped(mean_out, keep_ids, drop_ids)        # survivor -> [dropped]\n",
    "    if assign_topk and assign_topk > 1:\n"
    "        assign = _assign_dropped_soft(mean_out, keep_ids, drop_ids, topk=assign_topk)   # {i:[(j,frac)]}\n"
    "    else:\n"
    "        _hard = _assign_dropped(mean_out, keep_ids, drop_ids)                           # {i:[j]}\n"
    "        assign = {i: [(j, 1.0) for j in _hard[i]] for i in keep_ids}\n",
    src)

# 3. members + weight: carry per-member assignment fractions
src = repl(
    "        members = [i] + assign[i]\n"
    "        w = freq_l[members].clamp(min=1e-6) ** freq_exp\n",
    "        _pairs = [(i, 1.0)] + assign[i]\n"
    "        members = [p[0] for p in _pairs]\n"
    "        _fr = torch.tensor([p[1] for p in _pairs], device=freq_l.device, dtype=freq_l.dtype)\n"
    "        w = (freq_l[members].clamp(min=1e-6) ** freq_exp) * _fr\n",
    src)

# 4. fold_layer call site in phase_b_merge -> pass assign_topk
src = repl(
    "            mg, md = fold_layer(x, gu_t, dn_t, keep[li], drop[li], freq[li], device, stats,\n"
    "                                freq_exp=args.freq_exponent, norm_anchor=args.norm_anchor)\n",
    "            mg, md = fold_layer(x, gu_t, dn_t, keep[li], drop[li], freq[li], device, stats,\n"
    "                                freq_exp=args.freq_exponent, norm_anchor=args.norm_anchor,\n"
    "                                assign_topk=args.assign_topk)\n",
    src)

# 5. argparse
src = repl(
    '    ap.add_argument("--norm-anchor", default="survivor", choices=["survivor", "members_mean"])\n',
    '    ap.add_argument("--norm-anchor", default="survivor", choices=["survivor", "members_mean"])\n'
    '    ap.add_argument("--assign-topk", type=int, default=0)\n',
    src)

ast.parse(src)  # fail before writing if syntactically broken
open(path, "w").write(src)
print(f"patched {path}: {orig_len} -> {len(src)} bytes (ast OK)")
