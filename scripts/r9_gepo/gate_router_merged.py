#!/usr/bin/env python
"""Prove the run3 ROUTER LoRA actually landed in the merged weights.

WHY THIS GATE IS NEW FOR run3
-----------------------------
run1/run2 touched only nn.Linear modules, and merge_gepo1.py's existing gates
(22712 tensors, 19 mtp, config keys) are sufficient for that. run3 also trains the
MoE router, which the runtime exposes as a FUSED 3D nn.Parameter (mlp.gate.weight)
reached via peft target_parameters / lora.ParamWrapper rather than target_modules.

That difference is invisible to every existing gate: the router merges INTO AN
EXISTING TENSOR, so tensor count, mtp count and config are byte-identical whether or
not the router merged. A merge_and_unload that silently skipped the ParamWrapper
would yield a merged model that is run2-plus-a-dropout-change; the GGUF would build,
load and eval perfectly, and the router experiment would have evaporated with nothing
failing. peft 0.20.0 does implement ParamWrapper.merge, but "the library implements
it" is not evidence about THIS artifact.

KEYS ARE DISCOVERED, NOT SPELLED
--------------------------------
The first version of this gate hardcoded `model.layers.N.mlp.gate.weight` -- the
RUNTIME module path that the LoRA regex matches. The CHECKPOINT prefixes those keys
with `model.language_model.`, so every lookup missed and the gate reported 40/40
routers "missing" on a merge that had in fact succeeded. Keys are now found by
suffix against the base index, so the gate cannot fail that way again. The positive
control is what exposed it: a tensor named in the LoRA regex cannot be absent from
the base, so "POSITIVE missing" could only mean the gate was looking in the wrong
place.

FOUR POLARITIES -- a gate that can only pass is not a gate:
  ROUTER    layers.*.mlp.gate.weight   MUST differ    with --expect-router changed
                                        MUST be EQUAL  with --expect-router unchanged
                                        (run3 scope; run4 reverted to run2 scope, and
                                         asserting the routers did NOT move proves that
                                         rather than assuming it)
  POSITIVE  shared_expert / linear_attn MUST differ   (run2 scope; if identical, the
                                                       merge did nothing at all and a
                                                       "router differs" result would be
                                                       measuring something else)
  NEGATIVE  mlp.experts.*               MUST be EQUAL (never in any scope)
  MTP       mtp.*                       MUST be EQUAL (copied verbatim from base; its
                                                       router must NOT move)
"""
import json
import os
import sys

from safetensors import safe_open

BASE, MERGED = sys.argv[1], sys.argv[2]
# --expect-router changed|unchanged. run3 put LoRA on the 40 MoE routers, so they MUST
# move. run4 went back to run2 scope (ROUTER_LORA=0), so they must NOT -- and that is a
# real assertion, not a formality: it proves run4 is genuinely run2-scope and did not
# quietly inherit run3's scope. Default is "changed" so run3's build is unaffected.
EXPECT = "changed"
for i, a in enumerate(sys.argv):
    if a == "--expect-router" and i + 1 < len(sys.argv):
        EXPECT = sys.argv[i + 1]
if EXPECT not in ("changed", "unchanged"):
    print(f"REFUSE: --expect-router must be 'changed' or 'unchanged', got {EXPECT!r}")
    sys.exit(2)
ROUTER_MUST_MOVE = EXPECT == "changed"
N_LAYERS = 40


def index(d):
    return json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]


bw, mw = index(BASE), index(MERGED)


def get(d, wm, key):
    if key not in wm:
        return None
    with safe_open(os.path.join(d, wm[key]), framework="pt") as f:
        return f.get_tensor(key)


fails = []


def compare(key, must_differ, label):
    b, m = get(BASE, bw, key), get(MERGED, mw, key)
    if b is None or m is None:
        fails.append(f"{label}: {key} missing (base={b is not None} merged={m is not None})")
        return None
    if b.shape != m.shape:
        fails.append(f"{label}: {key} shape {tuple(b.shape)} -> {tuple(m.shape)}")
        return None
    d = (b.float() - m.float()).abs().max().item()
    if must_differ and d == 0:
        fails.append(f"{label}: {key} IDENTICAL to base but must have moved")
    if (not must_differ) and d > 0:
        fails.append(f"{label}: {key} CHANGED (max|delta|={d:.3e}) but must be untouched")
    return d


# --- discover, don't spell -----------------------------------------------------
routers = sorted(k for k in bw if k.endswith("mlp.gate.weight") and not k.startswith("mtp."))
mtp_routers = sorted(k for k in bw if k.endswith("mlp.gate.weight") and k.startswith("mtp."))
if len(routers) != N_LAYERS:
    print(f"REFUSE: found {len(routers)} non-MTP router keys, expected {N_LAYERS}")
    sys.exit(2)

print(f"=== ROUTER — all {len(routers)} must have "
      f"{'MOVED (run3 scope)' if ROUTER_MUST_MOVE else 'STAYED EQUAL (run2/run4 scope)'} ===")
deltas = []
for k in routers:
    d = compare(k, ROUTER_MUST_MOVE, "ROUTER")
    if d is not None:
        deltas.append(d)
if deltas:
    s = sorted(deltas)
    print(f"  moved: {sum(1 for d in deltas if d > 0)}/{len(routers)}")
    print(f"  max|delta|  min {s[0]:.3e}   p50 {s[len(s)//2]:.3e}   max {s[-1]:.3e}")

print("\n=== POSITIVE control (run2 scope) — must also have moved ===")
for suf in ("mlp.shared_expert.gate_proj.weight", "linear_attn.out_proj.weight"):
    k = next((x for x in sorted(bw) if x.endswith(suf) and not x.startswith("mtp.")), None)
    if k is None:
        fails.append(f"POSITIVE: no key ending {suf}")
        continue
    d = compare(k, True, "POSITIVE")
    if d is not None:
        print(f"  {k}\n      max|delta|={d:.3e}")

print("\n=== NEGATIVE control (never in scope) — must be untouched ===")
neg = [x for x in sorted(bw) if ".mlp.experts." in x][:2]
for k in neg:
    d = compare(k, False, "NEGATIVE")
    if d is not None:
        print(f"  {k}\n      max|delta|={d:.3e}")

print("\n=== MTP block — copied verbatim, its router must NOT move ===")
for k in mtp_routers[:2]:
    d = compare(k, False, "MTP")
    if d is not None:
        print(f"  {k}\n      max|delta|={d:.3e}")

print()
if fails:
    for f in fails:
        print(f"  FAIL {f}")
    print(f"\nROUTER_MERGE_GATE_FAIL ({len(fails)} problem(s))")
    sys.exit(1)
print("ROUTER_MERGE_GATE_OK")
