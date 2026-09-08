#!/usr/bin/env python3
"""Pin the serving geometry into humaneval_full_think.yaml.

WHY: this template pinned `max_gen_toks: 16384` but NO `llama_ctx`, so the per-slot
context was whatever the GPU planner happened to choose at launch. Two banked CoderX
HE+ cells were served at 16384 vs 24576 total and are therefore on different bases.
`llama_ctx` is the TOTAL pool llama-server splits across `--parallel` slots (bug-597),
so a total alone is not a geometry -- the required --parallel is documented here and
must be passed AND read back from server.log.

49152 / 2 slots = 24576 per slot > max_gen_toks 16384, leaving ~8k for the prompt.

Edits ONLY backend_args + a header comment. The frozen greedy `generation:` block is
not touched (canonical-sampler doctrine).
"""
import sys

P = sys.argv[1] if len(sys.argv) > 1 else \
    "/srv/ml/repos/omnimergekit/eval/templates/humaneval_full_think.yaml"

HEADER = """# HumanEval (164) in chat+thinking mode.
#
# SERVING GEOMETRY IS PINNED. `llama_ctx` below is the TOTAL context llama-server
# divides across `--parallel` slots -- it is NOT the per-slot size. This template is
# specified at **--parallel 2**: 49152 / 2 = 24576 per slot, which clears
# max_gen_toks 16384 with ~8k of prompt headroom.
#
# ALWAYS launch this template at `--parallel 2` and VERIFY the server logged
# `new slot ... n_ctx = 24576`. A larger --parallel silently shrinks the per-slot
# window below max_gen_toks and the run degrades without erroring (the T172.4
# SAT_COLLAPSE trap). Do NOT override llama_ctx via --metadata: that is how two
# CoderX cells ended up on 16384 vs 24576 and became incomparable.
#
# Sampler GREEDY (frozen). Do not edit `generation:` -- pass --sampler instead.
"""

src = open(P).read()
if "llama_ctx" in src:
    print("ALREADY PINNED -- refusing to double-patch")
    sys.exit(1)

# 1. header
if not src.startswith("#"):
    src = HEADER + src

# 2. llama_ctx into the existing backend_args block, as its first key
needle = "backend_args:\n  lm_eval_include_path:"
if needle not in src:
    print("REFUSE: backend_args block not in the expected shape")
    sys.exit(2)
src = src.replace(
    needle,
    "backend_args:\n"
    "  llama_ctx: 49152          # TOTAL; run at --parallel 2 => 24576/slot\n"
    "  lm_eval_include_path:",
    1,
)

open(P, "w").write(src)
print("PATCHED", P)

# 3. read it back and assert, rather than trusting the write
import yaml  # noqa: E402
d = yaml.safe_load(open(P))
ctx = d["backend_args"]["llama_ctx"]
mgt = d["generation"]["max_gen_toks"]
per_slot = ctx // 2
assert ctx == 49152, ctx
assert per_slot > mgt, f"per-slot {per_slot} <= max_gen_toks {mgt}"
assert d["generation"]["do_sample"] is False and d["generation"]["temperature"] == 0.0
assert d["n"] == 164 and len(d["selection"]["indices"]) == 164
print(f"VERIFY OK: llama_ctx={ctx} parallel=2 -> {per_slot}/slot > max_gen_toks={mgt}; "
      f"greedy intact; n=164")
