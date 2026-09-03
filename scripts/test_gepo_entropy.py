#!/usr/bin/env python
"""Polarity battery for GEPO-entropy advantage shaping (arXiv 2607.16850, Eq. 7-9).

Tests the REAL method source lifted out of gepo_trainer.py -- not a replica -- so the
test cannot drift from the implementation. A shaping rule that never fires, or fires on
the wrong sign, is indistinguishable from a correct one in the loss curve; run4 was lost
to exactly that class of invisibility, so every branch gets an explicit control.
"""
import ast
import json
import sys

import torch


def _load():
    """Bind the two real methods onto a stub with the attributes they touch."""
    src = open("scripts/gepo_trainer.py").read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GEPOTrainer")
    want = {"_maybe_shape_advantages_by_group_entropy", "_require_old_logps",
            "_tiers_of"}
    fns = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert len(fns) == 3, [f.name for f in fns]
    for f in fns:   # drop @staticmethod; we bind them by hand below
        f.decorator_list = []
    ns = {"torch": torch, "json": json}
    exec(compile(ast.Module(body=fns, type_ignores=[]), "<gepo>", "exec"), ns)

    class Stub:
        num_generations = 2
        gepo_entropy = True
        gepo_alpha_low = 0.5
        gepo_alpha_high = 0.2
        gepo_beta_low = 0.2
        gepo_beta_high = 0.3
        gepo_gamma = 0.01
        gepo_entropy_warmup = 1
        gepo_entropy_silent_limit = 25
        _gepo_H_low = None
        _gepo_H_high = None
        _require_old_logps = ns["_require_old_logps"]
        _maybe_shape_advantages_by_group_entropy = ns["_maybe_shape_advantages_by_group_entropy"]
        _tiers_of = staticmethod(ns["_tiers_of"])

    return Stub


Stub = _load()


def make(logps_per_group, advs):
    """logps_per_group: per-token logp for each group (constant within group)."""
    G = 2
    rows = []
    for lp in logps_per_group:
        rows += [[lp] * 4] * G
    return {
        "old_per_token_logps": torch.tensor(rows, dtype=torch.float32),
        "completion_mask": torch.ones(len(rows), 4),
        "advantages": torch.tensor(advs, dtype=torch.float32),
    }


P = F = 0


def chk(label, cond):
    global P, F
    print(("  ok   " if cond else "  FAIL ") + label)
    P, F = (P + 1, F) if cond else (P, F + 1)


# Groups: logp -1.0 -> H_g=1.0 (LOW), -2.0 -> H_g=2.0 (MID), -3.0 -> H_g=3.0 (HIGH)
# mu=2.0, sigma=0.8165 -> H_low=2-0.2*0.8165=1.837, H_high=2+0.3*0.8165=2.245
t = Stub()
out = make([-1.0, -2.0, -3.0], [+1.0, +1.0, +1.0, +1.0, +1.0, +1.0])
t._maybe_shape_advantages_by_group_entropy(out)
a = out["advantages"]
st = out["gepo_entropy_stats"]
print(f"thresholds: H_low={st['H_low']:.4f} H_high={st['H_high']:.4f} mu={st['H_mean']:.4f}")
chk("A  POSITIVE adv in LOW-entropy group  -> x alpha_low (0.5)", abs(a[0].item() - 0.5) < 1e-6)
chk("B  POSITIVE adv in MID-entropy group  -> UNCHANGED", abs(a[2].item() - 1.0) < 1e-6)
chk("C  POSITIVE adv in HIGH-entropy group -> UNCHANGED (asymmetry)", abs(a[4].item() - 1.0) < 1e-6)

t = Stub()
out = make([-1.0, -2.0, -3.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
t._maybe_shape_advantages_by_group_entropy(out)
a = out["advantages"]
chk("D  NEGATIVE adv in HIGH-entropy group -> x alpha_high (0.2)", abs(a[4].item() + 0.2) < 1e-6)
chk("E  NEGATIVE adv in MID-entropy group  -> UNCHANGED", abs(a[2].item() + 1.0) < 1e-6)
chk("F  NEGATIVE adv in LOW-entropy group  -> UNCHANGED (length-collapse guard)",
    abs(a[0].item() + 1.0) < 1e-6)

# H_g must be the per-TOKEN mean logp, so it is invariant to response length.
t = Stub()
o1 = make([-1.0, -2.0, -3.0], [1.0] * 6)
t._maybe_shape_advantages_by_group_entropy(o1)
t2 = Stub()
o2 = make([-1.0, -2.0, -3.0], [1.0] * 6)
o2["completion_mask"] = torch.cat([torch.ones(6, 2), torch.zeros(6, 2)], dim=1)
t2._maybe_shape_advantages_by_group_entropy(o2)
chk("G  H_g is per-token (halving length leaves thresholds unchanged)",
    abs(o1["gepo_entropy_stats"]["H_mean"] - o2["gepo_entropy_stats"]["H_mean"]) < 1e-5)

# EMA seeding: first call must ADOPT the batch estimate, not crawl up from 0.
t = Stub()
out = make([-1.0, -2.0, -3.0], [1.0] * 6)
t._maybe_shape_advantages_by_group_entropy(out)
chk("H  EMA seeded from first batch (not 0)", abs(float(t._gepo_H_high) - 2.2449) < 1e-3)
prev = float(t._gepo_H_high)
out2 = make([-5.0, -6.0, -7.0], [1.0] * 6)   # a large shift
t._maybe_shape_advantages_by_group_entropy(out2)
moved = float(t._gepo_H_high) - prev
chk("I  after seeding the EMA moves only ~gamma of the gap", 0 < moved < 0.06)

# Disabled must be a strict no-op (run4 reproducibility).
t = Stub(); t.gepo_entropy = False
out = make([-1.0, -2.0, -3.0], [1.0] * 6)
before = out["advantages"].clone()
t._maybe_shape_advantages_by_group_entropy(out)
chk("J  gepo_entropy=False is a strict no-op", torch.equal(out["advantages"], before)
    and "gepo_entropy_stats" not in out)

# The paper constraint must be enforced, not assumed.
t = Stub(); t.gepo_alpha_high = 0.9   # >= alpha_low
out = make([-1.0, -2.0, -3.0], [1.0] * 6)
try:
    t._maybe_shape_advantages_by_group_entropy(out)
    chk("K  alpha_high >= alpha_low REFUSED", False)
except ValueError:
    chk("K  alpha_high >= alpha_low REFUSED", True)

# Firing rate must be reported, so a silent no-op run is detectable at step 5.
t = Stub()
out = make([-1.0, -2.0, -3.0], [1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
t._maybe_shape_advantages_by_group_entropy(out)
st = out["gepo_entropy_stats"]
chk("L  firing rates reported and non-zero when shaping applies",
    st["frac_pos_attenuated"] > 0 and st["frac_neg_attenuated"] > 0 and st["frac_any_shaped"] > 0)

# --- M: per-tier breakdown SEPARATES a minority tier from the pool ------------
# run4's failure mode: 128/849 rows carry lambda>0, so a pool-wide rate is blind to
# whether the tier under study was touched at all.
t = Stub()
o = make([-1.0, -2.0, -3.0], [1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
t._maybe_shape_advantages_by_group_entropy(o, ["lcb_exec/T", "mbpp_exec/N", "mbpp_exec/N"])
pt = o["gepo_entropy_stats"]["per_tier"]
chk("M  per-tier splits minority tier from the pool",
    set(pt) == {"lcb_exec/T", "mbpp_exec/N"}
    and pt["lcb_exec/T"] == {"n": 1, "shaped": 1, "H_mean": 1.0, "frac_groups_shaped": 1.0}
    and pt["mbpp_exec/N"]["n"] == 2 and pt["mbpp_exec/N"]["shaped"] == 1
    and abs(pt["mbpp_exec/N"]["H_mean"] - 2.5) < 1e-6)

# --- N: a pool-wide rate would have HIDDEN the tier that matters ---------------
# Positive control for M. IDENTICAL entropies and advantages to M -- only the tier
# LABELS move, so frac_any_shaped is identical BY CONSTRUCTION while the tier we
# actually care about goes from fired (M) to never-touched (N). That is precisely
# run4's blindness, reproduced in two lines.
t = Stub()
o2 = make([-1.0, -2.0, -3.0], [1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
t._maybe_shape_advantages_by_group_entropy(o2, ["mbpp_exec/N", "lcb_exec/T", "mbpp_exec/N"])
chk("N  pooled rate is blind where per-tier is not",
    abs(o2["gepo_entropy_stats"]["frac_any_shaped"]
        - o["gepo_entropy_stats"]["frac_any_shaped"]) < 1e-9
    and o["gepo_entropy_stats"]["per_tier"]["lcb_exec/T"]["shaped"] == 1
    and o2["gepo_entropy_stats"]["per_tier"]["lcb_exec/T"]["shaped"] == 0)

# --- O: _tiers_of parses the JSON meta column, both row layouts ----------------
rows_p = [{"meta": json.dumps({"reward_kind": "lcb_exec"})},
          {"meta": json.dumps({"reward_kind": "mbpp_exec", "think": False})}]
chk("O  _tiers_of reads meta at prompt granularity",
    Stub._tiers_of(rows_p, 4, 2) == ["lcb_exec/T", "mbpp_exec/N"])
chk("O2 _tiers_of reads meta already expanded to rollouts",
    Stub._tiers_of([rows_p[0]] * 2 + [rows_p[1]] * 2, 4, 2) == ["lcb_exec/T", "mbpp_exec/N"])

# --- P: diagnostics can NEVER kill a run --------------------------------------
chk("P  _tiers_of returns None on junk instead of raising",
    Stub._tiers_of([{"nope": 1}, {"nope": 2}], 4, 2) is None
    and Stub._tiers_of(None, 4, 2) is None
    and Stub._tiers_of([{"meta": "{not json"}] * 2, 4, 2) is None)
t = Stub()
o3 = make([-1.0, -2.0, -3.0], [1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
t._maybe_shape_advantages_by_group_entropy(o3, None)
chk("P2 shaping still works with tiers=None (no per_tier key)",
    "per_tier" not in o3["gepo_entropy_stats"]
    and abs(float(o3["advantages"][0]) - 0.5) < 1e-6)

# --- Q: the silent-no-op detector fires, and RESETS on a live step -------------
# run4 ran 96h doing nothing measurable. A rule that shapes zero rollouts forever
# must stop the run, and must NOT stop one that is merely quiet for a while.
def silent_run(n, limit=3, live_at=None):
    t = Stub(); t.gepo_entropy_silent_limit = limit
    # Seed from a LIVE batch so the seeding call is itself a firing step and the
    # silent streak below is exactly `n`. (With warmup=1 the seed shapes in the same
    # call, so seeding from a quiet batch would silently cost one off the budget.)
    t._maybe_shape_advantages_by_group_entropy(
        make([-1.0, -2.0, -3.0], [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]), None)
    for i in range(n):
        live = live_at is not None and i == live_at
        o = (make([-1.0, -2.0, -3.0], [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]) if live
             else make([-2.0] * 3, [0.0] * 6))
        t._maybe_shape_advantages_by_group_entropy(o, None)
    return t


try:
    silent_run(3, limit=3)
    chk("Q  silent-no-op detector RAISES at the limit", False)
except RuntimeError as e:
    chk("Q  silent-no-op detector RAISES at the limit", "shaped NOTHING" in str(e))
try:
    silent_run(2, limit=3)
    chk("Q2 detector does NOT fire below the limit", True)
except RuntimeError:
    chk("Q2 detector does NOT fire below the limit", False)
try:
    silent_run(5, limit=3, live_at=2)   # 2 quiet, 1 live, 2 quiet -> never 3 in a row
    chk("Q3 a live step RESETS the silent counter", True)
except RuntimeError:
    chk("Q3 a live step RESETS the silent counter", False)
try:
    silent_run(50, limit=0)
    chk("Q4 limit=0 disables the detector", True)
except RuntimeError:
    chk("Q4 limit=0 disables the detector", False)

print(f"GEPO_ENTROPY_GATE pass={P} fail={F}")
sys.exit(1 if F else 0)
