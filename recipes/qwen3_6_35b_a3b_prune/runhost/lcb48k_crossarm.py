#!/usr/bin/env python
"""Cross-arm overlap on the 48k LCB basis -- the checks equal aggregates can hide.

Two questions outstanding:
 Q1. armD and base256e both solve 47. THE SAME 47? If the 184e cut solves a materially
     different set than the unpruned 256e, "the prune costs nothing on hard LCB" is far
     weaker than the score table implies.
 Q2. armI (57) and armJ (56) both tower over armD (47). Do they gain the SAME problems?
     A shared gain set = a systematic capability the hybrid recipe confers.
     Disjoint gains = two lucky draws that happen to land on the same count.
Q2 is what decides whether the hybrid result is a finding or a coincidence.
"""
import json
import os

R = "/srv/ml/eval_results/ream_arms/lcb_v6_77q_48k"
ARMS = [
    ("armI p12", "lcb48k_armI_hybrid_p12"),
    ("armJ p24", "lcb48k_armJ_hybrid_p24"),
    ("armE", "lcb48k_armE_reapsal_nomerge"),
    ("armF rnorm", "lcb48k_armF_rnorm_nomerge"),
    ("armD", "lcb48k_armD_ourssal_nomerge"),
    ("base256e", "lcb48k_base256e"),
    ("armB", "lcb48k_armB_reapsal_merge"),
    ("armG gs2", "lcb48k_armG_merge_gs2"),
    ("pub184e", "lcb48k_pub184e"),
]


def solved(cell):
    p = os.path.join(R, cell, "lcb_result.samples.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("task_id")
            if t is not None:
                out[t] = bool(d.get("passed"))
    return out


S, present = {}, []
for lab, cell in ARMS:
    sv = solved(cell)
    if len(sv) < 77:
        print("%-11s SKIP (incomplete: %d/77 rows -- still running)" % (lab, len(sv)))
        continue
    if sv:
        S[lab] = sv
        present.append(lab)
        n = sum(sv.values())
        print("%-11s solved=%-3d /%d" % (lab, n, len(sv)))
    else:
        print("%-11s (pending)" % lab)

if len(present) < 2:
    raise SystemExit

tasks = set.intersection(*(set(S[a]) for a in present))
print("\nshared task_ids: %d" % len(tasks))

# ---- Q1 ----
if "armD" in S and "base256e" in S:
    d, b = S["armD"], S["base256e"]
    both = [t for t in tasks if d[t] and b[t]]
    donly = [t for t in tasks if d[t] and not b[t]]
    bonly = [t for t in tasks if b[t] and not d[t]]
    print("\nQ1 armD(184e cut) vs base256e(unpruned), both solve 47:")
    print("   both=%d  armD-only=%d  base-only=%d  neither=%d"
          % (len(both), len(donly), len(bonly), len(tasks) - len(both) - len(donly) - len(bonly)))
    print("   => %s" % ("IDENTICAL sets" if not donly and not bonly else
                        "SAME COUNT, DIFFERENT PROBLEMS (%d swapped each way)" % len(donly)))

# ---- Q2 ----
if "armI p12" in S and "armJ p24" in S and "armD" in S:
    i, j, d = S["armI p12"], S["armJ p24"], S["armD"]
    gi = set(t for t in tasks if i[t] and not d[t])
    gj = set(t for t in tasks if j[t] and not d[t])
    li = set(t for t in tasks if d[t] and not i[t])
    lj = set(t for t in tasks if d[t] and not j[t])
    print("\nQ2 hybrid gains over armD:")
    print("   armI gains %d, loses %d   |   armJ gains %d, loses %d"
          % (len(gi), len(li), len(gj), len(lj)))
    print("   gain sets: shared=%d  armI-only=%d  armJ-only=%d"
          % (len(gi & gj), len(gi - gj), len(gj - gi)))
    ov = len(gi & gj) / max(1, len(gi | gj))
    print("   Jaccard(gain_I, gain_J) = %.2f" % ov)
    print("   => %s" % ("SHARED gain set: systematic property of the hybrid recipe"
                        if ov >= 0.5 else
                        "LARGELY DISJOINT gains: same count, different problems -- treat "
                        "the +10 as draw-dependent, not a stable capability"))
    print("   shared gained: %s" % sorted(gi & gj)[:14])

# ---- full pairwise agreement ----
print("\npairwise |solved sets| overlap (Jaccard):")
print("%-11s %s" % ("", " ".join("%-9s" % a[:9] for a in present)))
for a in present:
    row = []
    for b in present:
        A = set(t for t in tasks if S[a][t])
        B = set(t for t in tasks if S[b][t])
        row.append("%-9.2f" % (len(A & B) / max(1, len(A | B))))
    print("%-11s %s" % (a, " ".join(row)))
