#!/usr/bin/env python3
"""patch_dern_knobs.py <redist_dern_eq11.py> — expose two DERN knobs without changing defaults.

Adds:
  --freq-exponent FLOAT  (default 1.0)  : freq-weight = freq**exp before normalize (1.0 = identity)
  --norm-anchor  {survivor,members_mean} (default survivor = current Eq.11 behavior)
                         members_mean = anchor the merged output norm to the freq-weighted mean of
                         ALL members' (survivor + assigned-dropped) ORIGINAL output norms, instead
                         of the survivor's alone.

Idempotent + asserts each replacement is unique (fails loud rather than silently corrupting).
Defaults reproduce the T203 baseline byte-for-byte at the algorithm level.
"""
import ast
import sys

path = sys.argv[1]
src = open(path).read()
orig_len = len(src)

if "--freq-exponent" in src and "norm_anchor" in src:
    print("already patched")
    sys.exit(0)


def repl(old, new, s):
    n = s.count(old)
    assert n == 1, f"expected 1 occurrence, found {n} of:\n---\n{old[:120]}\n---"
    return s.replace(old, new)


# 1. fold_layer signature
src = repl(
    "def fold_layer(x, gu_t, dn_t, keep_ids, drop_ids, freq_l, device, stats):",
    'def fold_layer(x, gu_t, dn_t, keep_ids, drop_ids, freq_l, device, stats, freq_exp=1.0, norm_anchor="survivor"):',
    src)

# 2. freq exponent on the per-member weight
src = repl(
    "        w = freq_l[members].clamp(min=1e-6)\n",
    "        w = freq_l[members].clamp(min=1e-6) ** freq_exp\n",
    src)

# 3. norm-anchor block (selectable survivor vs members_mean)
old_anchor = (
    "        # --- DERN Eq.11 norm-equalization, SURVIVOR-anchored ---\n"
    "        o_surv = _expert_out(x, gu_t[i].float(), dn_t[i].float())   # survivor original [T,H]\n"
    "        o_merg = _expert_out(x, mg, md)                             # merged [T,H]\n"
    "        ns = o_surv.norm(dim=-1).mean()\n"
    "        nm = o_merg.norm(dim=-1).mean().clamp(min=1e-6)\n"
    "        s = (ns / nm).item()\n"
)
new_anchor = (
    "        # --- DERN Eq.11 norm-equalization (anchor selectable) ---\n"
    '        if norm_anchor == "members_mean":\n'
    "            mns = torch.stack([_expert_out(x, gu_t[j].float(), dn_t[j].float()).norm(dim=-1).mean()\n"
    "                               for j in members])                      # [m] per-member original norm\n"
    "            ns = (w.to(mns.device) * mns).sum()                        # freq-weighted mean magnitude\n"
    "        else:                                                          # survivor (baseline)\n"
    "            ns = _expert_out(x, gu_t[i].float(), dn_t[i].float()).norm(dim=-1).mean()\n"
    "        o_merg = _expert_out(x, mg, md)                             # merged [T,H]\n"
    "        nm = o_merg.norm(dim=-1).mean().clamp(min=1e-6)\n"
    "        s = (ns / nm).item()\n"
)
src = repl(old_anchor, new_anchor, src)

# 4. fold_layer call site in phase_b_merge
src = repl(
    "            mg, md = fold_layer(x, gu_t, dn_t, keep[li], drop[li], freq[li], device, stats)\n",
    "            mg, md = fold_layer(x, gu_t, dn_t, keep[li], drop[li], freq[li], device, stats,\n"
    "                                freq_exp=args.freq_exponent, norm_anchor=args.norm_anchor)\n",
    src)

# 5. argparse additions
src = repl(
    '    ap.add_argument("--seq-max-tokens", type=int, default=8192)\n',
    '    ap.add_argument("--seq-max-tokens", type=int, default=8192)\n'
    '    ap.add_argument("--freq-exponent", type=float, default=1.0)\n'
    '    ap.add_argument("--norm-anchor", default="survivor", choices=["survivor", "members_mean"])\n',
    src)

ast.parse(src)  # fail before writing if syntactically broken
open(path, "w").write(src)
print(f"patched {path}: {orig_len} -> {len(src)} bytes (ast OK)")
