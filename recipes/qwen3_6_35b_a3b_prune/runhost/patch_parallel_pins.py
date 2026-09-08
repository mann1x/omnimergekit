#!/usr/bin/env python3
"""Pin `llama_parallel` alongside `llama_ctx` in the two code templates.

WHY. `llama_ctx` is the TOTAL context llama-server divides across slots (bug-597).
A template that pins only the total does not pin a geometry: gpu_planner picks
`parallel` from free VRAM, so the same template serves 16384/slot on one launch and
4096/slot on the next, and two cells become incomparable without either of them
being "wrong". omk_eval already honours `backend_args.llama_parallel` as the
template FORCE (CLI --parallel still wins), so pinning it makes the served geometry
a property of the template instead of a property of the host's spare VRAM.

  multipl_e_100        16384 total / 4 =  4096 per slot  (max_gen_toks 1024)
  humaneval_full_think 49152 total / 2 = 24576 per slot  (max_gen_toks 16384)

Nothing else is touched -- in particular the frozen greedy `generation:` blocks.
"""
import sys
import yaml

TDIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/srv/ml/repos/omnimergekit/eval/templates"

WANT = {
    "multipl_e_100.yaml": (4, 16384, 4096),
    "humaneval_full_think.yaml": (2, 49152, 24576),
}

for fn, (par, total, per_slot) in WANT.items():
    p = f"{TDIR}/{fn}"
    src = open(p).read()
    if "llama_parallel" in src:
        print(f"SKIP {fn}: already pins llama_parallel")
        continue
    needle = "  llama_ctx: "
    if needle not in src:
        print(f"REFUSE {fn}: no llama_ctx line to anchor on")
        sys.exit(2)
    line_start = src.index(needle)
    line_end = src.index("\n", line_start) + 1
    ins = (f"  llama_parallel: {par}"
           f"           # FORCED: {total}/{par} = {per_slot} per slot\n")
    src = src[:line_end] + ins + src[line_end:]
    open(p, "w").write(src)
    print(f"PATCHED {fn}")

# read back and assert, rather than trusting the writes
print()
for fn, (par, total, per_slot) in WANT.items():
    d = yaml.safe_load(open(f"{TDIR}/{fn}"))
    ba = d["backend_args"]
    g = d["generation"]
    assert ba["llama_parallel"] == par, (fn, ba.get("llama_parallel"))
    assert ba["llama_ctx"] == total, (fn, ba.get("llama_ctx"))
    assert ba["llama_ctx"] // ba["llama_parallel"] == per_slot
    assert per_slot > g["max_gen_toks"], (fn, per_slot, g["max_gen_toks"])
    assert g["do_sample"] is False and g["temperature"] == 0.0, fn
    tb = int(g.get("thinking_token_budget", 0) or 0)
    # plan_ctx would AUTO-BUMP the total if parallel*(budget+headroom) exceeded it;
    # assert here that the pin survives the planner instead of discovering it in a log.
    if tb:
        assert par * (tb + 4096) <= total, (fn, "pin would be auto-bumped", par * (tb + 4096))
    print(f"VERIFY {fn}: ctx={total} parallel={par} -> {per_slot}/slot "
          f"> max_gen_toks={g['max_gen_toks']}; think_budget={tb}; greedy intact")
