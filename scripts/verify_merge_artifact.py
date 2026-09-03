#!/usr/bin/env python3
"""Verify a merged model against its base. Usage: verify_merge_artifact.py <base> <out> <skip_csv>

A STANDALONE FILE, not a heredoc. bug-660: in build_v6.sh this ran as
    "$PY" - args 2>&1 | tee -a "$L" <<'PYV' ... PYV
where the heredoc binds to the LAST pipeline command (tee), so tee printed the script
INTO the log, python got nothing, and the caller's `grep -q V6_VERIFY_OK "$L"` matched
the literal string inside the echoed source. The gate passed by reading its own code
and the merged model went unverified. [[feedback_verify_artifacts_not_exitcodes]]
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open

base, out = sys.argv[1], sys.argv[2]
skip = [p for p in sys.argv[3].split(",") if p]


def index(d):
    return json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]


bi, oi = index(base), index(out)
rc = 0
print(f"  tensors base={len(bi)} out={len(oi)} identical_set={set(bi) == set(oi)}")
if set(bi) != set(oi):
    print(f"    MISSING: {sorted(set(bi)-set(oi))[:8]}")
    print(f"    EXTRA  : {sorted(set(oi)-set(bi))[:8]}")
    rc = 1
for lab, rx in (("mtp", r"mtp|nextn|eh_proj"), ("vision", r"visual|vision")):
    nb = len([n for n in bi if re.search(rx, n, re.I)])
    no = len([n for n in oi if re.search(rx, n, re.I)])
    print(f"  {lab:7} base={nb} out={no} {'ok' if nb == no else 'MISMATCH'}")
    rc |= (nb != no)


def get(d, idx, name):
    with safe_open(os.path.join(d, idx[name]), framework="pt") as h:
        return h.get_tensor(name)


for pat in skip:
    names = [n for n in oi if pat in n][:3]
    if not names:
        print(f"  skip '{pat}': NO TENSOR MATCHED — pattern inert!")
        rc = 1
        continue
    bad = [n for n in names if not torch.equal(get(base, bi, n), get(out, oi, n))]
    print(f"  skip '{pat}': {len(names)} sampled, {len(bad)} differ from base "
          f"{'ok' if not bad else 'FAIL ' + str(bad[:2])}")
    rc |= bool(bad)

# NO-OP CONTROL: without this, every "identical" above is satisfied by copying the base.
ctrl = [n for n in oi if "self_attn.q_proj.weight" in n and not any(p in n for p in skip)]
if ctrl:
    n = ctrl[0]
    b, o = get(base, bi, n), get(out, oi, n)
    changed = not torch.equal(b, o)
    rel = (b.float() - o.float()).norm() / b.float().norm().clamp(min=1e-9)
    print(f"  CONTROL {n}: changed={changed} rel_L2={rel:.4f} "
          f"{'ok' if changed else 'FAIL — merge was a NO-OP'}")
    rc |= (not changed)
else:
    print("  CONTROL: no non-skipped attn tensor found")
    rc = 1

# SECOND CONTROL: the merged model must differ from the TASK-BASE too, else the output
# is just Qwen3.6 wearing a 3.8 label.
tb = os.environ.get("TASK_BASE")
if tb and os.path.exists(os.path.join(tb, "model.safetensors.index.json")):
    ti = index(tb)
    n = ctrl[0] if ctrl else None
    if n:
        d38 = (get(base, bi, n).float() - get(out, oi, n).float()).norm()
        d36 = (get(tb, ti, n).float() - get(out, oi, n).float()).norm()
        print(f"  ANCHORING {n}: ||out-3.8||={d38:.2f}  ||out-3.6||={d36:.2f} "
              f"-> closer to {'3.8 base (correct)' if d38 < d36 else '3.6 TASK-BASE (WRONG)'}")
        rc |= (d38 >= d36)
print("V6_VERIFY_OK" if rc == 0 else "V6_VERIFY_FAIL")
sys.exit(rc)
