#!/usr/bin/env python3
"""Selftest for grpo_reward_efficiency.

A reward whose failure modes are only DESCRIBED in a docstring is untested. Each case
here asserts a property the run depends on, and the budget-placement cases assert the
FAILING side too -- a check that only ever passes cannot tell you the budget is wrong.
[[feedback_a_check_gold_fails_is_a_broken_check]] [[feedback_a_zero_needs_a_nonzero_floor_control]]
"""
from __future__ import annotations

import statistics as st
import sys

from grpo_reward_efficiency import (
    FORMAT_BONUS,
    R_CORRECT,
    budget_from_lengths,
    make_efficiency_reward,
)


class FakeTok:
    """1 token per whitespace word. Deterministic, no model download."""
    def __call__(self, text, add_special_tokens=False):
        class Enc:
            input_ids = (text or "").split()
        return Enc()


MARKER_TOKS = 5  # "The correct answer is (C)"


def mk(nwords: int, correct: bool, letter: str = "C") -> str:
    """Emits EXACTLY nwords whitespace tokens, so a cap test is not off by one."""
    body = " ".join(["w"] * max(nwords - MARKER_TOKS, 1))
    return f"{body} The correct answer is ({letter if correct else 'A'})"


def run(lengths, correct, budget, lam, max_completion=100000, kind="mc_letter"):
    r = make_efficiency_reward(FakeTok(), None, max_completion)
    n = len(lengths)
    comps = [mk(L, c) for L, c in zip(lengths, correct)]
    metas = [{"reward_kind": kind, "length_lambda": lam,
              "length_budget": budget, "think": True} for _ in range(n)]
    return r(comps, prompts=["p"] * n, gold=["C"] * n, meta=metas)


FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


print("=== 1. ORDERING: worst passer must beat best failure ===")
rs = run([10, 5000, 20], [True, True, False], budget=1000, lam=0.9)
passers = [rs[0], rs[1]]
check("worst passer > best failure", min(passers) > rs[2],
      f"min_passer={min(passers):.3f} failure={rs[2]:.3f}")
check("failure is exactly 0.0", rs[2] == 0.0, f"got {rs[2]}")

print("\n=== 2. budget WAY BELOW distribution -> length term is CONSTANT (dead) ===")
lens = [420, 700, 1150, 1600, 2400, 7600]  # measured-shape, right-skewed
rs = run(lens, [True] * 4, budget=100, lam=0.5)
sd = st.pstdev(rs)
check("all passers clip to lenpen=1.0 -> zero spread", sd == 0.0,
      f"rewards={[round(x,3) for x in rs]} pstdev={sd:.6f}")
check("this is the SILENT death mode (cancels under scale_rewards=group)", sd == 0.0)

print("\n=== 3. budget WAY ABOVE distribution -> length term is negligible ===")
rs_hi = run(lens, [True] * 4, budget=10_000_000, lam=0.5)
sd_hi = st.pstdev(rs_hi)
check("spread is ~0 but NOT identically 0", 0.0 < sd_hi < 1e-3,
      f"pstdev={sd_hi:.3e}")

print("\n=== 4. budget at the P35 quantile (AN operating point) -> real spread ===")
budget = budget_from_lengths(lens, q=0.35)
rs_ok = run(lens, [True] * 4, budget=budget, lam=0.5)
sd_ok = st.pstdev(rs_ok)
check("spread is materially larger than the too-high case", sd_ok > 100 * sd_hi,
      f"budget={budget} pstdev={sd_ok:.4f} vs too_high={sd_hi:.3e}")
check("spread is materially larger than the too-low case", sd_ok > sd,
      f"{sd_ok:.4f} > {sd:.4f}")

print("\n=== 5. frac_under_budget discriminates the three placements ===")
def frac_under(budget):
    r = make_efficiency_reward(FakeTok(), None, 100000)
    metas = [{"reward_kind": "mc_letter", "length_lambda": 0.5,
              "length_budget": budget, "think": True} for _ in lens]
    r([mk(L, True) for L in lens], prompts=["p"] * len(lens),
      gold=["C"] * len(lens), meta=metas)
    b = list(r._state["byk"].values())[0]
    return b["under"] / b["pass"]
f_lo, f_ok, f_hi = frac_under(100), frac_under(budget), frac_under(10_000_000)
check("too-low budget -> frac_under_budget == 0.00", f_lo == 0.0, f"{f_lo:.2f}")
check("too-high budget -> frac_under_budget == 1.00", f_hi == 1.0, f"{f_hi:.2f}")
check("AN point -> strictly between", 0.0 < f_ok < 1.0, f"{f_ok:.2f}")

print("\n=== 6. REFUSALS ===")
try:
    run([100], [True], budget=None, lam=0.3)
    check("lambda>0 with no budget must REFUSE", False, "no exception raised")
except ValueError as e:
    check("lambda>0 with no budget must REFUSE", "length_budget" in str(e))
try:
    run([100], [True], budget=None, lam=0.0)
    check("lambda==0 with no budget is FINE (replay)", True)
except Exception as e:
    check("lambda==0 with no budget is FINE (replay)", False, repr(e))
try:
    run([100], [True], budget=100, lam=0.3, kind="nonesuch")
    check("unknown reward_kind must REFUSE", False, "no exception raised")
except ValueError as e:
    check("unknown reward_kind must REFUSE", "no verifier registered" in str(e))
try:
    run([100], [True], budget=100, lam=0.3, kind="lcb_exec")
    check("lcb_exec with no verifier must REFUSE", False, "no exception raised")
except ValueError as e:
    check("lcb_exec with no verifier must REFUSE", "lcb_verifier" in str(e))

print("\n=== 7. CENSORED rollout is never credited as a passer ===")
rs = run([50], [True], budget=1000, lam=0.5, max_completion=50)
# mk(50) emits exactly 50 tokens, so nt >= max_completion holds.
check("rollout at the completion cap scores 0.0", rs[0] == 0.0, f"got {rs[0]}")

print("\n=== 8. replay tier (lambda=0) is pure correctness ===")
rs = run([10, 9000], [True, True], budget=None, lam=0.0)
check("both passers identical regardless of length",
      rs[0] == rs[1] == R_CORRECT + FORMAT_BONUS,
      f"{[round(x,3) for x in rs]}")

print()
if FAILS:
    print(f"SELFTEST: FAIL ({len(FAILS)}): {', '.join(FAILS)}")
    sys.exit(1)
print("SELFTEST: PASS")
