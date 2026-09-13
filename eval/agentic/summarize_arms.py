#!/usr/bin/env python3
"""summarize_arms.py — aggregate agentic-bench arms with the comparability guards.

Reads inspect_ai `.eval` logs and produces a per-arm table plus a PAIRED
comparison. Guards, in the order they can invalidate a delta:

1. BASIS   — every arm's STACK.txt must differ ONLY on the declared variable.
             Reported, not assumed. Diff it yourself before believing a number.
2. CAP     — if one arm hits max_tokens materially more than its sibling, the
             accuracy delta is partly a LENGTH measurement, not a quality one.
             This is reported next to every delta and gates the verdict.
3. PAIRED  — arms see the same sample ids, so the comparison is McNemar on the
             discordant pairs, NOT two independent proportions. Using an
             unpaired test here throws away the pairing and widens the CI.
4. ERRORS  — a sample that errored is not a sample that failed. Counted apart;
             a delta computed over different denominators is not a delta.

Usage:
    summarize_arms.py <armA_logdir> <armB_logdir> [...]
"""
from __future__ import annotations

import sys
import pathlib
from collections import defaultdict

try:
    from inspect_ai.log import read_eval_log
except ImportError:
    sys.exit("FAIL: inspect_ai not importable — use the agentic env's python "
             "(see eval/agentic/setup_agentic_env.sh)")


def latest_eval(d: pathlib.Path) -> pathlib.Path | None:
    logs = sorted(d.glob("*.eval"))
    return logs[-1] if logs else None


def cap_hits(sample) -> int:
    """Count model calls that stopped on the token cap.

    Walks events so MULTI-TURN calls are counted, not just the final output --
    a mid-loop cap is exactly the failure mode this guard exists to catch, and
    it never appears in sample.output.
    """
    n = 0
    for ev in (getattr(sample, "events", None) or []):
        out = getattr(ev, "output", None)
        if out is not None and getattr(out, "stop_reason", None) == "max_tokens":
            n += 1
    if n == 0:
        out = getattr(sample, "output", None)
        if out is not None and getattr(out, "stop_reason", None) == "max_tokens":
            n = 1
    return n


def out_tokens(sample) -> int:
    tot = 0
    mu = getattr(sample, "model_usage", None) or {}
    for u in mu.values():
        tot += getattr(u, "output_tokens", 0) or 0
    return tot


def load_arm(d: pathlib.Path):
    f = latest_eval(d)
    if f is None:
        return None
    log = read_eval_log(str(f))
    rows = {}
    for s in (log.samples or []):
        val = None
        for sc in (s.scores or {}).values():
            val = sc.value
            break
        correct = (val == "C") or (val is True) or (val == 1) or (val == 1.0)
        rows[str(s.id)] = {
            "correct": bool(correct),
            "error": getattr(s, "error", None) is not None,
            "caps": cap_hits(s),
            "otok": out_tokens(s),
            "steps": sum(1 for m in (s.messages or []) if getattr(m, "role", "") == "assistant"),
        }
    stack = d / "STACK.txt"
    return {"dir": d, "file": f, "rows": rows,
            "stack": stack.read_text() if stack.exists() else None}


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial test on the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main(argv):
    if len(argv) < 1:
        sys.exit(__doc__)
    arms = {}
    for d in argv:
        p = pathlib.Path(d)
        a = load_arm(p)
        if a is None:
            print(f"WARN: no .eval log in {p} — skipping")
            continue
        arms[p.name] = a
    if not arms:
        sys.exit("FAIL: no arms loaded")

    print(f"{'arm':28} {'n':>4} {'acc':>7} {'err':>4} {'cap-hits':>9} "
          f"{'out_tok':>10} {'tok/correct':>12} {'steps/samp':>10}")
    print("-" * 92)
    for name, a in arms.items():
        r = a["rows"]
        n = len(r)
        ok = sum(v["correct"] for v in r.values())
        err = sum(v["error"] for v in r.values())
        caps = sum(v["caps"] for v in r.values())
        otok = sum(v["otok"] for v in r.values())
        steps = sum(v["steps"] for v in r.values())
        acc = ok / n if n else 0.0
        tpc = (otok / ok) if ok else float("nan")
        print(f"{name:28} {n:4d} {acc:7.3f} {err:4d} {caps:9d} "
              f"{otok:10d} {tpc:12.0f} {steps/n if n else 0:10.1f}")

    names = list(arms)
    if len(names) != 2:
        return 0

    A, B = arms[names[0]], arms[names[1]]
    shared = sorted(set(A["rows"]) & set(B["rows"]))
    print(f"\n=== PAIRED comparison on {len(shared)} shared samples "
          f"({names[0]} vs {names[1]}) ===")
    if not shared:
        print("  no shared sample ids — NOT comparable")
        return 0

    b = sum(1 for k in shared if A["rows"][k]["correct"] and not B["rows"][k]["correct"])
    c = sum(1 for k in shared if not A["rows"][k]["correct"] and B["rows"][k]["correct"])
    both = sum(1 for k in shared if A["rows"][k]["correct"] and B["rows"][k]["correct"])
    neither = len(shared) - both - b - c
    p = mcnemar(b, c)
    accA = (both + b) / len(shared)
    accB = (both + c) / len(shared)
    print(f"  both correct {both:4d} | both wrong {neither:4d} | "
          f"only {names[0]} {b:4d} | only {names[1]} {c:4d}")
    print(f"  acc {names[0]}={accA:.3f}  {names[1]}={accB:.3f}  "
          f"delta={accB - accA:+.3f}pp-frac   McNemar exact p={p:.4f}")

    capA = sum(A["rows"][k]["caps"] for k in shared)
    capB = sum(B["rows"][k]["caps"] for k in shared)
    print(f"\n  CAP GUARD: cap-hits {names[0]}={capA}  {names[1]}={capB}")
    hi, lo = max(capA, capB), min(capA, capB)
    # A ratio alone is not a confound: 1-vs-0 is infinite ratio and zero
    # consequence. Require a MATERIAL absolute count (>=1% of samples) before
    # calling the delta length-confounded, or the guard cries wolf on noise.
    material = hi >= max(5, 0.01 * len(shared))
    if material and (lo == 0 or hi / max(lo, 1) >= 1.5):
        print("  *** CAP ASYMMETRY *** one arm hits the token cap materially more.")
        print("      The accuracy delta is CONFOUNDED with generation length.")
        print("      Re-run with a higher cap, or report this as a length result.")
    elif hi > 0:
        print(f"  cap-hits present but immaterial ({hi} of {len(shared)} samples) "
              "— not a length artefact.")
    else:
        print("  no cap-hits in either arm — not a length artefact.")

    errA = sum(A["rows"][k]["error"] for k in shared)
    errB = sum(B["rows"][k]["error"] for k in shared)
    if errA or errB:
        print(f"  ERROR GUARD: errored samples {names[0]}={errA} {names[1]}={errB} "
              "(errored != failed; check before quoting the delta)")

    if A["stack"] is None or B["stack"] is None:
        print("\n  *** NO STACK.txt on one or both arms — basis UNVERIFIED (§1.4.5) ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
