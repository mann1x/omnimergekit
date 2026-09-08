#!/usr/bin/env python3
"""Idempotently insert the MPE sampler forwarding (--temperature/--top-p/--top-k)
into dispatch_multipl's gen_cmd of a target omk_eval.py, preserving everything
else (bs2 is ahead of the local checkout — MRCR/DCA/gpu-plan). Anchor is the
dispatch_multipl --max-tokens → --limit pair, unique to that gen_cmd."""
import sys

f = sys.argv[1]
s = open(f).read()
if '"--temperature", str(g.get("temperature"' in s:
    print("already patched — no-op")
    sys.exit(0)

anchor = (
    '                "--max-tokens", str(max_tokens),\n'
    '                "--limit", str(0 if problems_map else (n if n > 0 else 0)),'
)
ins = (
    '                "--max-tokens", str(max_tokens),\n'
    '                # Sampler from generation.* — defaults greedy (0.0/1.0/0) so\n'
    '                # frozen canonical MPE templates are byte-identical; shadow\n'
    '                # templates carry the gemma vendor sampler. min_p/repeat_penalty\n'
    '                # stay server-launch flags (generator never sends them).\n'
    '                "--temperature", str(g.get("temperature", 0.0)),\n'
    '                "--top-p", str(g.get("top_p", 1.0)),\n'
    '                "--top-k", str(g.get("top_k", 0)),\n'
    '                "--limit", str(0 if problems_map else (n if n > 0 else 0)),'
)
if anchor not in s:
    print("ANCHOR NOT FOUND — dispatch_multipl gen_cmd shape changed; patch manually")
    sys.exit(2)
if s.count(anchor) != 1:
    print(f"ANCHOR NOT UNIQUE ({s.count(anchor)} matches) — aborting")
    sys.exit(3)
open(f, "w").write(s.replace(anchor, ins, 1))
print("patched OK")
