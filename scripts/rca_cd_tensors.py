#!/usr/bin/env python3
"""RCA: why do v7 CD-mix GGUFs ruminate? Diff per-tensor quant types of each
CD tier against the healthy plain Q4_K_M, spotlighting termination-critical
tensors (output / token_embd / norms) and the CD skip-list (attn_v / ffn_down /
token_embd / router=ffn_gate_inp)."""
import re
from collections import Counter
from gguf import GGUFReader

GD = "/mnt/sdc/ml/quant_sweep_gguf"
PLAIN = f"{GD}/gemma-4-A4B-98e-v7-coder-it-Q4_K_M.gguf"
CDS = [
    f"{GD}/gemma-4-A4B-98e-v7-coder-it-CD-Q4_K_M.gguf",
    f"{GD}/gemma-4-A4B-98e-v7-coder-it-CD-IQ4_K_M.gguf",
]

def types(p):
    r = GGUFReader(p)
    return {t.name: t.tensor_type.name for t in r.tensors}

plain = types(PLAIN)
print("=== PLAIN Q4_K_M type histogram ===")
print("  ", dict(Counter(plain.values())))

CRIT = ["output.weight", "token_embd.weight", "output_norm.weight"]
SKIP = ["attn_v", "ffn_down", "token_embd", "ffn_gate_inp", "attn_k", "attn_q", "attn_output", "ffn_gate", "ffn_up"]

for cdpath in CDS:
    name = cdpath.split("/")[-1]
    cd = types(cdpath)
    print(f"\n========== {name} ==========")
    print("  type histogram:", dict(Counter(cd.values())))
    print("  CRITICAL singletons:")
    for k in CRIT:
        pv, cv = plain.get(k, "-"), cd.get(k, "-")
        flag = "  <-- CHANGED" if pv != cv else ""
        print(f"    {k:22s} plain={pv:12s} CD={cv}{flag}")
    diffs = [(n, plain[n], cd[n]) for n in plain if n in cd and plain[n] != cd[n]]
    print(f"  {len(diffs)}/{len(plain)} tensors differ. change histogram (plain->CD):")
    for (a, b), n in sorted(Counter((a, b) for _, a, b in diffs).items(), key=lambda x: -x[1]):
        print(f"    {a:12s} -> {b:12s}: {n}")
    print("  differing by tensor class:")
    byc = {}
    for n, a, b in diffs:
        c = re.sub(r"blk\.\d+\.", "blk.N.", n)
        byc.setdefault(c, set()).add(f"{a}->{b}")
    for c in sorted(byc):
        print(f"    {c:36s}: {sorted(byc[c])}")
    print("  SKIP-LIST check (these should stay high-bit, NOT CD-aggressive):")
    for key in SKIP:
        ps = dict(Counter(v for k, v in plain.items() if key in k))
        cs = dict(Counter(v for k, v in cd.items() if key in k))
        flag = "  <-- DEGRADED" if ps != cs else ""
        print(f"    {key:14s} plain={ps}  CD={cs}{flag}")
