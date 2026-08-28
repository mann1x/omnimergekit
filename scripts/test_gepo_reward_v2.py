#!/usr/bin/env python3
"""Gold-anchored gate for the run4 reward. Asserts the properties training depends on.

A reward test that only checks "it returned a float" checks nothing. Each arm below
asserts a property whose violation would silently ruin a 20-hour run:

  1  ORDERING      worst passer > best failure. If this inverts, the model learns that
                   being wrong-and-short beats right-and-long.
  2  MONOTONE      shorter passer > longer passer, strictly.
  3  VARIANCE      the length share of within-group variance is much larger than v1's
                   measured 11.3%. This is the ENTIRE point of v2 -- GRPO normalises by
                   group std, so a term owning 11% of the variance owns 11% of the
                   gradient.
  4  SCALE-FREE    a group whose lengths are 10x smaller gets a comparable length
                   signal. v1's signal decayed as the model converged; v2's must not.
  5  CONVERGED     when lengths genuinely agree, the MAD floor damps the signal, so
                   noise is not amplified into a gradient.
  6  TRUNCATION    a censored rollout does not set the length scale, and clipped
                   rollouts cannot be read as "short".
  7  ALL-FAIL      a group with no passers yields all zeros (no length info exists).
  8  REPLAY        lambda=0 rows score pure correctness 1/0 -- no length pressure.
  9  REFUSE        an unregistered reward_kind RAISES rather than scoring 0.0.
"""
from __future__ import annotations

import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gepo_reward_v2 import make_gepo_reward_v2  # noqa: E402

MAXC = 12288
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        fails += 1


class FakeTok:
    """1 token per 4 chars -- deterministic, so lengths in these arms are exact."""
    def __call__(self, text, add_special_tokens=False):
        class R:
            input_ids = [0] * (len(text) // 4)
        return R()


def fake_lcb(passing_texts):
    def v(text, meta):
        return (text in passing_texts), "ok"
    return v


def build(passing, base=0.6, alpha=0.4):
    return make_gepo_reward_v2(FakeTok(), fake_lcb(passing), MAXC, base=base, alpha=alpha)


def rollouts(lengths):
    """Distinct texts of the requested token lengths (4 chars per token)."""
    return [f"{i:03d}" + "x" * (4 * n - 3) for i, n in enumerate(lengths)]


def v1(ntok, lam=0.7, budget=12288):
    return 1.0 - lam * min(ntok / budget, 1.0)


# ---------------------------------------------------------------- arms 1-3
print("=== arms 1-3: ordering, monotonicity, variance share (5 of 8 pass) ===")
lens = [5200, 5800, 6100, 6600, 7000, 6000, 6000, 6000]
texts = rollouts(lens)
passing = set(texts[:5])
r = build(passing)
out = r(completions=texts, prompts=["p"] * 8, meta=[{"tests": [1]}] * 8)
pas = [out[i] for i in range(5)]
fai = [out[i] for i in range(5, 8)]
print("   rewards:", [f"{x:.3f}" for x in out])
check("1 ordering: worst passer > best failure", min(pas) > max(fai),
      f"min_pass={min(pas):.3f} max_fail={max(fai):.3f}")
check("2 monotone: shortest > longest passer", pas[0] > pas[4],
      f"{pas[0]:.3f} > {pas[4]:.3f}")

v1_grp = [v1(n) for n in lens[:5]] + [0.0, 0.0, 0.0]
v1_share = st.pstdev(v1_grp[:5]) / st.pstdev(v1_grp) * 100
v2_share = st.pstdev(pas) / st.pstdev(out) * 100
check("3 variance share beats v1 substantially", v2_share > 3 * v1_share,
      f"v1={v1_share:.1f}%  v2={v2_share:.1f}%")

# ---------------------------------------------------------------- arm 4
print("\n=== arm 4: scale-free (same shape, 10x shorter) ===")
small = [520, 580, 610, 660, 700, 600, 600, 600]
ts = rollouts(small)
r2 = build(set(ts[:5]))
o2 = r2(completions=ts, prompts=["p"] * 8, meta=[{"tests": [1]}] * 8)
share_small = st.pstdev(o2[:5]) / st.pstdev(o2) * 100
v1_small = [v1(n) for n in small[:5]] + [0.0, 0.0, 0.0]
v1_share_small = st.pstdev(v1_small[:5]) / st.pstdev(v1_small) * 100
check("4 v2 length signal survives 10x shorter lengths",
      abs(share_small - v2_share) < 5.0,
      f"v2 {v2_share:.1f}% -> {share_small:.1f}%  (v1 {v1_share:.1f}% -> {v1_share_small:.1f}%)")

# ---------------------------------------------------------------- arm 5
print("\n=== arm 5: converged lengths => damped signal (MAD floor) ===")
conv = [6000, 6001, 6000, 6001, 6000, 9000, 9000, 9000]
tc = rollouts(conv)
r3 = build(set(tc[:5]))
o3 = r3(completions=tc, prompts=["p"] * 8, meta=[{"tests": [1]}] * 8)
check("5 near-identical lengths give a small spread", st.pstdev(o3[:5]) < 0.05,
      f"passer std={st.pstdev(o3[:5]):.4f}")

# ---------------------------------------------------------------- arm 6
print("\n=== arm 6: truncated rollouts are censored, not 'short' ===")
tl = [4000, 4200, 4400, MAXC, MAXC, 5000, 5000, 5000]
tt = rollouts(tl)
r4 = build(set(tt[:5]))          # the two capped rollouts "pass" -- worst case
o4 = r4(completions=tt, prompts=["p"] * 8, meta=[{"tests": [1]}] * 8)
print("   rewards:", [f"{x:.3f}" for x in o4])
check("6a capped passer scores STRICTLY below every uncapped passer",
      max(o4[3], o4[4]) < min(o4[0], o4[1], o4[2]),
      f"capped={max(o4[3], o4[4]):.3f} min_uncapped={min(o4[0], o4[1], o4[2]):.3f}")
check("6c capped passer still beats a failure",
      min(o4[3], o4[4]) > max(o4[5], o4[6], o4[7]),
      f"capped={min(o4[3], o4[4]):.3f} fail={max(o4[5], o4[6], o4[7]):.3f}")
check("6b capped rollouts did not set the scale (short one still best)",
      o4[0] == max(o4), f"o4[0]={o4[0]:.3f} max={max(o4):.3f}")

# ---------------------------------------------------------------- arm 7
print("\n=== arm 7: all-fail group => all zeros ===")
r5 = build(set())
o5 = r5(completions=rollouts([5000] * 4), prompts=["p"] * 4, meta=[{"tests": [1]}] * 4)
check("7 no passers => no gradient invented", all(x == 0.0 for x in o5), str(o5))

# ---------------------------------------------------------------- arm 8
print("\n=== arm 8: replay rows (lambda=0) score pure correctness ===")
rt = ["The correct answer is (C)", "The correct answer is (A)",
      "The correct answer is (C)", "no answer here"]
r6 = build(set())     # lcb verifier must never be consulted for mc_letter
o6 = r6(completions=rt, prompts=["q"] * 4,
        meta=[{"reward_kind": "mc_letter", "length_lambda": 0.0}] * 4,
        gold=["C"] * 4)
check("8a correct letters -> 1.0", o6[0] == 1.0 and o6[2] == 1.0, str(o6))
check("8b wrong / missing -> 0.0", o6[1] == 0.0 and o6[3] == 0.0, str(o6))
check("8c no length pressure: both correct rows score identically",
      o6[0] == o6[2], f"{o6[0]} vs {o6[2]}")

# ---------------------------------------------------------------- arm 9
print("\n=== arm 9: unregistered reward_kind must REFUSE, not score 0.0 ===")
r7 = build(set())
try:
    r7(completions=["x"], prompts=["p"], meta=[{"reward_kind": "not_wired"}])
    check("9 raises on unknown reward_kind", False, "returned instead of raising")
except ValueError as e:
    check("9 raises on unknown reward_kind", "REFUSE" in str(e), str(e)[:70])

print(f"\n{'REWARD_V2_OK' if fails == 0 else f'REWARD_V2_FAIL ({fails} failing)'}")
sys.exit(1 if fails else 0)
