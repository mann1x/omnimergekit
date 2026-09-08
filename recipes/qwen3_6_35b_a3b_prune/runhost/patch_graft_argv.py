#!/usr/bin/env python3
"""Let graft_mtp.py take arm names on argv instead of only its hardcoded five.

armG/armH need the same published-184e MTP block grafted in as every other arm -- held
CONSTANT across arms so it can never explain a difference between them. The existing script
already does exactly that; the only thing stopping reuse is that ARMS is a module constant.
Default is unchanged, so re-running it bare still means the original five.

Idempotent: refuses to double-patch.
"""
import sys

P = "/mnt/sdc/ream-work/graft_mtp.py"
OLD = 'ARMS = ["armD_ourssal_nomerge", "armC", "armB", "armE", "armF_rnorm_nomerge"]\n'
NEW = ('ARMS = sys.argv[1:] or ["armD_ourssal_nomerge", "armC", "armB", "armE",\n'
       '                        "armF_rnorm_nomerge"]\n')

s = open(P).read()
if "sys.argv[1:]" in s:
    print("already patched -- no change")
    sys.exit(0)
if OLD not in s:
    sys.exit("FAIL: ARMS line not found verbatim; refusing to guess")
open(P, "w").write(s.replace(OLD, NEW, 1))

back = open(P).read()
if "sys.argv[1:]" not in back:
    sys.exit("FAIL: patch absent after write")
# `import sys` must already be present -- main() calls sys.exit().
if "\nimport sys\n" not in back:
    sys.exit("FAIL: sys not imported; patch would NameError at import time")
print("patched: ARMS = sys.argv[1:] or [<original five>]")
