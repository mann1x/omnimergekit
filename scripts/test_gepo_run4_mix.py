#!/usr/bin/env python3
"""Integration gate for the run4 mixed pool: BOTH tiers must produce reward variance.

The failure this exists to catch is silent by construction. GRPO's gradient is the
WITHIN-GROUP spread of the reward; a tier whose rollouts all score the same value
contributes exactly nothing, and it looks completely normal in the loss curve, the
grad_norm, and the reward mean. A replay tier that scored 0.0 for every rollout --
because its reward_kind was unwired, or its `gold` column was missing, or its
verifier never matched -- would burn a third of a 20-hour run producing no gradient
while dragging the policy.

So this asserts, on the REAL pools with a stub model:
  1  every pool row carries a reward_kind that the v2 dispatch actually handles
  2  every mc_letter row carries a gold answer (else it is dead by construction)
  3  the LCB tier produces non-zero within-group reward variance
  4  the mc_letter tier produces non-zero within-group reward variance
  5  the mbpp_exec tier produces non-zero within-group reward variance,
     verified by EXECUTING the pool's own reference solution -- a verifier that
     rejects the known-correct answer is broken, and would zero the whole tier
  6  replay rows carry no length pressure: two correct replay rollouts of very
     different lengths score identically
  7  LCB rows DO carry length pressure (the tiers must not be swapped)
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from gepo_reward_v2 import make_gepo_reward_v2, verify_mbpp_exec  # noqa: E402

LCB = REPO / "eval" / "lcb" / "lcb_rl_pool.jsonl"
REPLAY = REPO / "eval" / "replay" / "gepo_replay_pool.jsonl"
KNOWN = {"lcb_exec", "mc_letter", "mbpp_exec"}
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        fails += 1


def load(p):
    if not p.is_file():
        sys.exit(f"REFUSE: missing pool {p}")
    return [json.loads(x) for x in p.open() if x.strip()]


lcb, rep = load(LCB), load(REPLAY)
rows = lcb + rep
for r in lcb:
    r.setdefault("meta", {}).setdefault("reward_kind", "lcb_exec")
print(f"pools: lcb={len(lcb)} replay={len(rep)} total={len(rows)}")

print("\n=== arm 1-2: every row is dispatchable and scorable ===")
kinds = {}
for r in rows:
    k = r["meta"].get("reward_kind", "MISSING")
    kinds[k] = kinds.get(k, 0) + 1
check("1 all reward_kinds are wired", set(kinds) <= KNOWN, str(kinds))
mc = [r for r in rep if r["meta"]["reward_kind"] == "mc_letter"]
check("2 every mc_letter row has a gold answer",
      all(str(r.get("gold") or "").strip() in list("ABCD") for r in mc),
      f"{len(mc)} rows")


class Tok:
    def __call__(self, t, add_special_tokens=False):
        class R:
            input_ids = [0] * (len(t) // 4)
        return R()


def stub_lcb(passing):
    return lambda text, meta: ((text in passing), "ok")


print("\n=== arm 3: LCB tier has within-group variance ===")
texts = [f"{i:03d}" + "x" * (4 * n - 3) for i, n in enumerate([4000, 5000, 6000, 7000])]
r_lcb = make_gepo_reward_v2(Tok(), stub_lcb(set(texts[:3])), 12288)
o = r_lcb(completions=texts, prompts=["p"] * 4,
          meta=[json.dumps(lcb[0]["meta"])] * 4, gold=[""] * 4)
check("3 LCB group spread > 0", st.pstdev(o) > 0, f"{[f'{x:.3f}' for x in o]}")

print("\n=== arm 4: mc_letter tier has within-group variance ===")
g = mc[0]["gold"]
wrong = "ABCD".replace(g, "")[0]
mct = [f"The correct answer is ({g})", f"The correct answer is ({wrong})",
       f"The correct answer is ({g})", "I am not sure"]
r_mc = make_gepo_reward_v2(Tok(), stub_lcb(set()), 12288)
o = r_mc(completions=mct, prompts=["q"] * 4,
         meta=[json.dumps(mc[0]["meta"])] * 4, gold=[g] * 4)
check("4 mc_letter group spread > 0", st.pstdev(o) > 0, f"{o}")

print("\n=== arm 5: mbpp_exec verifier accepts the pool's OWN reference solution ===")
mb = [r for r in rep if r["meta"]["reward_kind"] == "mbpp_exec"]
ok_ref = sum(1 for r in mb[:25]
             if verify_mbpp_exec(r["meta"]["reference_code"], r["meta"]))
check("5a reference solutions pass their own tests", ok_ref >= 23,
      f"{ok_ref}/25 (a verifier that rejects gold would zero the tier)")
o = make_gepo_reward_v2(Tok(), stub_lcb(set()), 12288)(
    completions=[mb[0]["meta"]["reference_code"], "def wrong(): return None",
                 mb[0]["meta"]["reference_code"], "syntax ((("],
    prompts=["m"] * 4, meta=[json.dumps(mb[0]["meta"])] * 4, gold=[""] * 4)
check("5b mbpp_exec group spread > 0", st.pstdev(o) > 0, f"{o}")

print("\n=== arm 6-7: length pressure is on LCB and NOT on replay ===")
short = mb[0]["meta"]["reference_code"]
longr = short + "\n" + "# padding comment\n" * 400
o = make_gepo_reward_v2(Tok(), stub_lcb(set()), 12288)(
    completions=[short, longr], prompts=["m"] * 2,
    meta=[json.dumps(mb[0]["meta"])] * 2, gold=[""] * 2)
check("6 replay: very different lengths, identical reward", o[0] == o[1],
      f"short={o[0]} long={o[1]}")
lt = [f"{i:03d}" + "x" * (4 * n - 3) for i, n in enumerate([3000, 9000])]
o = make_gepo_reward_v2(Tok(), stub_lcb(set(lt)), 12288)(
    completions=lt, prompts=["p"] * 2,
    meta=[json.dumps(lcb[0]["meta"])] * 2, gold=[""] * 2)
check("7 LCB: shorter scores strictly higher", o[0] > o[1], f"{o}")

print(f"\n{'RUN4_MIX_OK' if fails == 0 else f'RUN4_MIX_FAIL ({fails} failing)'}")
sys.exit(1 if fails else 0)
