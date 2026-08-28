#!/usr/bin/env python3
"""Gate for the REBALANCED mixed pool (eval/replay/gepo_mixed_pool.jsonl) and the
per-row thinking machinery that makes it trainable.

The pool this guards is one file carrying four tiers -- lcb_exec/T, mbpp_exec/N,
mbpp_exec/T, mc_letter/N -- where the OLD shape was two files (an all-thinking LCB
pool plus an all-no-think replay pool). Every assumption that made the two-file shape
safe has to be re-established for the one-file shape, and each one fails silently:

  * `think` is a PER-ROW property now. A row missing it inherits a default, and the
    default is "think if lcb_exec" -- correct for the old pools, and wrong for the 100
    mbpp thinking rows, which would render no-think and quietly become duplicates of
    rows already in the pool.
  * The mbpp thinking and no-think slices must be DISJOINT problem sets. The same
    problem in both is not two data points; it is one problem at double weight, and
    it makes the /T-vs-/N comparison a paired measurement that nothing accounts for.
  * length_lambda must be 0 on everything except LCB. A replay row with length
    pressure is not defending capability, it is adding to the brevity signal that
    already cost -12.99pp on LCB in run3.
  * The difficulty filter now runs on the ASSEMBLED pool. Its tier-wipeout guard has
    to key on the /T|/N tier, not the bare reward_kind, or erasing all 100 mbpp
    thinking rows leaves `mbpp_exec` looking healthy at 371.

Arms 5-8 drive apply_difficulty directly, because a guard that has never been made to
fire is not a guard. [[feedback_a_check_gold_fails_is_a_broken_check]]
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import gepo_brevity as gb  # noqa: E402

POOL = REPO / "eval" / "replay" / "gepo_mixed_pool.jsonl"
KNOWN = {"lcb_exec", "mc_letter", "mbpp_exec"}
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        fails += 1


def refuses(fn, *a):
    """True when fn REFUSES (SystemExit with a message), False when it returns."""
    try:
        fn(*a)
        return False, "returned"
    except SystemExit as e:
        return True, str(e)[:70]


if not POOL.is_file():
    sys.exit(f"REFUSE: missing pool {POOL}")
rows = [json.loads(x) for x in POOL.open() if x.strip()]


def tier(r):
    k = r["meta"].get("reward_kind", "?")
    return k + ("/T" if r["meta"].get("think", k == "lcb_exec") else "/N")


cen: dict[str, int] = {}
for r in rows:
    cen[tier(r)] = cen.get(tier(r), 0) + 1
print(f"pool: {len(rows)} rows  {dict(sorted(cen.items()))}")

print("\n=== arms 1-4: the pool as shipped ===")
check("1 every row declares think explicitly",
      all("think" in r["meta"] for r in rows),
      f"{sum(1 for r in rows if 'think' not in r['meta'])} rows missing it")
check("2 every reward_kind is wired",
      {r["meta"].get("reward_kind") for r in rows} <= KNOWN,
      str(sorted({r["meta"].get("reward_kind") for r in rows})))
mbT = {r["id"].split("#")[0] for r in rows
       if r["meta"]["reward_kind"] == "mbpp_exec" and r["meta"]["think"]}
mbN = {r["id"].split("#")[0] for r in rows
       if r["meta"]["reward_kind"] == "mbpp_exec" and not r["meta"]["think"]}
check("3 mbpp think/no-think slices are DISJOINT problems", not (mbT & mbN),
      f"|T|={len(mbT)} |N|={len(mbN)} overlap={len(mbT & mbN)}")
bad_lam = [r["id"] for r in rows
           if (float(r["meta"].get("length_lambda", 0.0)) != 0.0)
           != (r["meta"]["reward_kind"] == "lcb_exec")]
check("4 length pressure is on LCB and ONLY on LCB", not bad_lam,
      f"{len(bad_lam)} offending rows")
mc = [r for r in rows if r["meta"]["reward_kind"] == "mc_letter"]
check("4b every mc_letter row carries gold",
      all(str(r.get("gold") or "").strip() in list("ABCD") for r in mc),
      f"{len(mc)} rows")
check("4c ids are unique", len({r["id"] for r in rows}) == len(rows),
      f"{len({r['id'] for r in rows})} of {len(rows)}")

print("\n=== arms 5-8: apply_difficulty, driven until each guard FIRES ===")
G = 8
TMP = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "gepo_mixed_pool_test"
TMP.mkdir(parents=True, exist_ok=True)


def prof_file(d, name):
    f = TMP / f"{name}.json"
    f.write_text(json.dumps(d))
    return str(f)


# 5  A partial profile must KEEP every unprofiled row and drop only the measured ones.
half = rows[:40]
p5 = {r["id"]: {"G": G, "unanimous": i % 2 == 0} for i, r in enumerate(half)}
out = gb.apply_difficulty([dict(r) for r in rows], prof_file(p5, "p5"), G, str(POOL))
check("5 keeps unprofiled, drops only measured-unanimous",
      len(out) == len(rows) - 20,
      f"{len(out)} of {len(rows)} (expected {len(rows) - 20})")

# 6  A profile measured at a DIFFERENT G must be refused: unanimity is a property OF
#    the group size, so a G=4 profile says nothing about a G=8 run.
p6 = {r["id"]: {"G": 4, "unanimous": False} for r in half}
fired, msg = refuses(gb.apply_difficulty, [dict(r) for r in rows],
                     prof_file(p6, "p6"), G, str(POOL))
check("6 REFUSES a profile measured at another G", fired, msg)

# 7  A profile describing NO row of this pool must be refused rather than silently
#    filtering nothing -- that is the shape of a wrong --replay-difficulty path.
p7 = {"not/a/real/id#N": {"G": G, "unanimous": True}}
fired, msg = refuses(gb.apply_difficulty, [dict(r) for r in rows],
                     prof_file(p7, "p7"), G, str(POOL))
check("7 REFUSES a profile that describes no row of this pool", fired, msg)

# 8  THE tier-wipeout guard, aimed at the slice it was rewritten for: mark every
#    mbpp_exec THINKING row unanimous and nothing else. Under a bare-reward_kind key
#    mbpp_exec still shows 371 survivors and the guard stays silent; under the /T|/N
#    tier key it must fire.
p8 = {r["id"]: {"G": G, "unanimous": True} for r in rows if tier(r) == "mbpp_exec/T"}
fired, msg = refuses(gb.apply_difficulty, [dict(r) for r in rows],
                     prof_file(p8, "p8"), G, str(POOL))
check("8 REFUSES when the filter erases the mbpp THINKING slice alone", fired, msg)

print(f"\n{'MIXED_POOL_OK' if fails == 0 else f'MIXED_POOL_FAIL ({fails} failing)'}")
sys.exit(1 if fails else 0)
