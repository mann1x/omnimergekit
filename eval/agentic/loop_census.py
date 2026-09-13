#!/usr/bin/env python3
"""loop_census.py — measure LOOPING, not accuracy, across agentic-bench arms.

The reasoning-re-injection hypothesis is about DEGENERATION: re-fed prior
thoughts prime the model to keep ruminating until it repeats verbatim and/or
runs to the token budget. Tool-call correctness is a downstream proxy that only
moves when a loop happens to wreck a task — so accuracy is the wrong endpoint
and a null there says little. This measures the endpoint directly.

Detectors (the ones this project already uses for rumination):
  REPEAT  r"(.{2,40}?)\1{7,}"  — a 2-40 char unit repeated >=8x  (verbatim loop)
  RUNCHAR r"(.)\1{40,}"        — one char repeated >=41x
  RUNAWAY reasoning length above a threshold (default 10k chars)
  FORCED  llama.cpp had to inject the end-of-thinking tag (budget exhausted)
          == a termination failure, counted from the SERVER log separately.

Reported per arm and, when exactly two arms are given, as a PAIRED comparison
over shared sample ids (McNemar exact), because both arms see the same samples.

Usage:
    loop_census.py <armA_logdir> <armB_logdir> [--runaway-chars N]
"""
from __future__ import annotations

import re
import sys
import pathlib
import statistics as st

try:
    from inspect_ai.log import read_eval_log
except ImportError:
    sys.exit("FAIL: use the agentic env python (eval/agentic/setup_agentic_env.sh)")

RE_REPEAT = re.compile(r"(.{2,40}?)\1{7,}", re.S)
RE_RUNCHAR = re.compile(r"(.)\1{40,}", re.S)


def reasoning_parts(sample):
    for m in (sample.messages or []):
        if getattr(m, "role", "") != "assistant":
            continue
        c = getattr(m, "content", None)
        if isinstance(c, list):
            for p in c:
                if getattr(p, "type", "") == "reasoning":
                    yield getattr(p, "reasoning", "") or ""


def census(d: pathlib.Path, runaway: int):
    logs = sorted(d.glob("*.eval"))
    if not logs:
        return None
    log = read_eval_log(str(logs[-1]))
    rows = {}
    for s in (log.samples or []):
        texts = list(reasoning_parts(s))
        lens = [len(t) for t in texts]
        n_rep = sum(1 for t in texts if RE_REPEAT.search(t))
        n_run = sum(1 for t in texts if RE_RUNCHAR.search(t))
        n_away = sum(1 for L in lens if L >= runaway)
        val = None
        for sc in (s.scores or {}).values():
            val = sc.value
            break
        rows[str(s.id)] = {
            "turns": len(texts),
            "rep_turns": n_rep,
            "run_turns": n_run,
            "away_turns": n_away,
            # a sample is "looped" if ANY of its turns degenerated
            "looped": bool(n_rep or n_run or n_away),
            "maxlen": max(lens) if lens else 0,
            "lens": lens,
            "correct": (val == "C") or (val is True) or (val == 1) or (val == 1.0),
        }
    return rows


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def main(argv):
    runaway = 10000
    if "--runaway-chars" in argv:
        i = argv.index("--runaway-chars")
        runaway = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if not argv:
        sys.exit(__doc__)

    arms = {}
    for a in argv:
        p = pathlib.Path(a)
        r = census(p, runaway)
        if r is None:
            print(f"WARN: no .eval in {p}")
            continue
        arms[p.name] = r

    print(f"runaway threshold = {runaway} chars\n")
    print(f"{'arm':22} {'n':>4} {'LOOPED':>7} {'rate':>7} {'rep_t':>6} {'run_t':>6} "
          f"{'away_t':>7} {'turns':>6} {'p50':>6} {'p90':>7} {'max':>8}")
    print("-" * 100)
    for name, rows in arms.items():
        n = len(rows)
        looped = sum(v["looped"] for v in rows.values())
        allen = [L for v in rows.values() for L in v["lens"]]
        turns = sum(v["turns"] for v in rows.values())
        p50 = int(st.median(allen)) if allen else 0
        p90 = int(sorted(allen)[int(.9 * len(allen)) - 1]) if allen else 0
        print(f"{name:22} {n:4d} {looped:7d} {looped/max(n,1):7.3f} "
              f"{sum(v['rep_turns'] for v in rows.values()):6d} "
              f"{sum(v['run_turns'] for v in rows.values()):6d} "
              f"{sum(v['away_turns'] for v in rows.values()):7d} "
              f"{turns:6d} {p50:6d} {p90:7d} {max(allen) if allen else 0:8d}")

    names = list(arms)
    if len(names) != 2:
        return 0
    A, B = arms[names[0]], arms[names[1]]
    shared = sorted(set(A) & set(B))
    b = sum(1 for k in shared if A[k]["looped"] and not B[k]["looped"])
    c = sum(1 for k in shared if not A[k]["looped"] and B[k]["looped"])
    both = sum(1 for k in shared if A[k]["looped"] and B[k]["looped"])
    print(f"\n=== PAIRED LOOP comparison on {len(shared)} shared samples ===")
    print(f"  both looped {both} | only {names[0]} {b} | only {names[1]} {c} "
          f"| neither {len(shared)-both-b-c}")
    print(f"  loop rate {names[0]}={(both+b)/len(shared):.3f}  "
          f"{names[1]}={(both+c)/len(shared):.3f}  McNemar exact p={mcnemar(b,c):.4f}")
    # does looping actually cost accuracy? the link the hypothesis assumes
    for nm, R in ((names[0], A), (names[1], B)):
        lp = [v for v in R.values() if v["looped"]]
        cl = [v for v in R.values() if not v["looped"]]
        if lp and cl:
            print(f"  {nm}: acc when looped={sum(v['correct'] for v in lp)/len(lp):.3f} "
                  f"(n={len(lp)}) vs clean={sum(v['correct'] for v in cl)/len(cl):.3f} "
                  f"(n={len(cl)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
