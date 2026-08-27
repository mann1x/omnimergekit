#!/usr/bin/env python3
"""Control-verify the replay pool's contamination gates by POISONING the pool.

The clean build prints `GPQA Diamond ids in pool: 0` / `MBPP scored/fewshot: 0` /
`HumanEval solutions: 0`. Three zeros prove nothing on their own -- a gate that
looks in the wrong place, compares the wrong key, or silently gets an empty holdout
set also prints three zeros. This asserts each gate FIRES on a row it must catch,
and that the clean pool is clean, which is the pair that makes the zeros readable.

Each arm injects ONE real holdout row, taken from the actual source file rather
than fabricated, so a gate cannot pass by rejecting a malformed probe.

Run:  python scripts/test_gepo_replay_gate.py
"""
from __future__ import annotations

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from build_gepo_replay_pool import (GPQA_SNAP, HE_SNAP, MBPP_SNAP,  # noqa: E402
                                    MBPP_FEWSHOT_IDS, build_gpqa, build_mbpp,
                                    check_holdout, code_hash)

fails = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails += 1


print("building the clean pool ...")
gpqa, _ = build_gpqa(0)
mbpp, _ = build_mbpp(1)

print("\n=== arm 0: the clean pool must pass all three gates ===")
g, m, h = check_holdout(gpqa, mbpp)
check("clean GPQA gate silent", not g, f"leaks={len(g)}")
check("clean MBPP gate silent", not m, f"leaks={len(m)}")
check("clean HumanEval gate silent", not h, f"leaks={len(h)}")
check("pool is non-empty", len(gpqa) > 0 and len(mbpp) > 0,
      f"gpqa={len(gpqa)} mbpp={len(mbpp)}")

print("\n=== arm 1: a real GPQA Diamond row must be CAUGHT ===")
dia = list(csv.DictReader((pathlib.Path(GPQA_SNAP) / "gpqa_diamond.csv")
                          .open(encoding="utf-8")))
poison_g = dict(gpqa[0])
poison_g["id"] = f"gpqa_main/{dia[0]['Record ID']}"
g, m, h = check_holdout(gpqa + [poison_g], mbpp)
check("GPQA gate fires", len(g) == 1, f"caught={g}")
check("GPQA poison does not disturb the other gates", not m and not h)

print("\n=== arm 2: a real MBPP scored-test row must be CAUGHT ===")
test = pd.read_parquet(pathlib.Path(MBPP_SNAP) / "full" / "test-00000-of-00001.parquet")
poison_m = dict(mbpp[0])
poison_m["id"] = f"mbpp/{int(test['task_id'].iloc[0])}"
g, m, h = check_holdout(gpqa, mbpp + [poison_m])
check("MBPP test-split gate fires", len(m) == 1, f"caught={m}")

print("\n=== arm 3: a pinned 3-shot exemplar must be CAUGHT ===")
poison_f = dict(mbpp[0])
poison_f["id"] = f"mbpp/{sorted(MBPP_FEWSHOT_IDS)[0]}"
g, m, h = check_holdout(gpqa, mbpp + [poison_f])
check("MBPP fewshot-exemplar gate fires", len(m) == 1, f"caught={m}")

print("\n=== arm 4: a real HumanEval solution must be CAUGHT ===")
he = pd.read_parquet(pathlib.Path(HE_SNAP) / "test-00000-of-00001.parquet")
poison_h = dict(mbpp[0])
poison_h["id"] = "mbpp/999999"
poison_h["meta"] = dict(mbpp[0]["meta"])
poison_h["meta"]["code_sha256"] = code_hash(str(he["canonical_solution"].iloc[0]))
g, m, h = check_holdout(gpqa, mbpp + [poison_h])
check("HumanEval overlap gate fires", len(h) == 1, f"caught={h}")

print("\n=== arm 5: the holdout sets must be non-empty ===")
# A gate whose holdout set silently loaded as empty passes every clean build.
check("Diamond holdout non-empty", len(dia) == 198, f"n={len(dia)}")
check("MBPP test holdout non-empty", len(test) == 500, f"n={len(test)}")
check("HumanEval holdout non-empty", len(he) == 164, f"n={len(he)}")

print("\n=== arm 6: every replay row is lambda=0 with a known verifier ===")
allrows = gpqa + mbpp
check("all length_lambda == 0.0",
      all(r["meta"]["length_lambda"] == 0.0 for r in allrows))
check("all reward_kind known",
      all(r["meta"]["reward_kind"] in {"mc_letter", "mbpp_exec"} for r in allrows))
check("mc_letter rows carry a gold letter",
      all(r["gold"] in "ABCD" and len(r["gold"]) == 1
          for r in allrows if r["meta"]["reward_kind"] == "mc_letter"))
check("mbpp_exec rows carry >=1 test",
      all(r["meta"]["n_tests"] >= 1
          for r in allrows if r["meta"]["reward_kind"] == "mbpp_exec"))

print(f"\n{'REPLAY_GATE_OK' if fails == 0 else f'REPLAY_GATE_FAIL ({fails} failing)'}")
sys.exit(1 if fails else 0)
