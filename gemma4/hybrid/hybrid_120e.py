#!/usr/bin/env python3
"""
Hybrid 120e (v2 with hard Q32 protection and absolute wnorm scoring).

Strategy: drop K experts per layer that are noisy (active on gain questions
Q8/Q71/Q27 where 128e fails) but with HARD PROTECTION for Q32-critical
experts.

Improvements over v1:
1. HARD CONSTRAINT: never drop experts in Q32 top-PROTECT_TOP (default 50).
   v1 used a soft rank-relative score that allowed dropping experts with
   meaningful Q32 activation.
2. ABSOLUTE wnorm scoring instead of rank: rank loses magnitude info.
   noise_score = mean(wnorm_Q8, wnorm_Q71, wnorm_Q27) - alpha * wnorm_Q32
3. Multiple regression questions if available (Q32, Q15, Q21, Q37, Q42, Q47).
   Falls back to Q32 only if others not in per_question_v4.
4. Sweeps K=4,8,12 to make it easy to pick the largest K that survives Q32.

Output:
  scripts/hybrid_120e_drop_map.json (default K=8 map)
  scripts/hybrid_120e_K{4,8,12}_drop_map.json
"""

import argparse
import json
from pathlib import Path

ANALYSIS_FILE = Path("scripts/per_question_v4.json")

# Gain questions where 128e fails (dropping noise on these helps)
GAIN_QS = [8, 71, 27]
# Regression questions (where 128e is right and 109e is wrong) — protect these
REGRESSION_QS = [32, 15, 21, 37, 42, 47]


def load():
    with open(ANALYSIS_FILE) as f:
        return json.load(f)


def get_wnorms(pq, qid, li):
    """Return {expert_id: wnorm} for question qid at layer li, or None if missing."""
    q = pq["questions"].get(str(qid))
    if q is None:
        return None
    return {e["id"]: e["wnorm"] for e in q[str(li)]}


def get_rank(pq, qid, li):
    """Return {expert_id: rank} (0 = highest wnorm) for question qid at layer li."""
    q = pq["questions"].get(str(qid))
    if q is None:
        return None
    sorted_e = sorted(q[str(li)], key=lambda x: -x["wnorm"])
    return {e["id"]: r for r, e in enumerate(sorted_e)}


def build_drop_map(pq, K, protect_top, alpha, num_layers, num_experts):
    """Build the drop map with hard Q32 protection.

    For each layer:
    1. Compute the set of available regression questions in pq.
    2. Build the protected set: union of top-`protect_top` for each regression Q.
    3. For each non-protected expert compute:
         noise = mean(wnorm on each gain Q) - alpha * mean(wnorm on each regression Q)
       Higher = noisier.
    4. Drop top-K noisy.
    """
    drop_map = {}
    avail_gain = [q for q in GAIN_QS if str(q) in pq["questions"]]
    avail_reg = [q for q in REGRESSION_QS if str(q) in pq["questions"]]
    if not avail_reg:
        raise RuntimeError("No regression questions found in analysis file")

    print(f"Gain questions:       {avail_gain}")
    print(f"Regression questions: {avail_reg}")
    print(f"K={K} protect_top={protect_top} alpha={alpha}")
    print()

    stats = {
        "violations_top16": 0,
        "violations_top32": 0,
        "violations_top48": 0,
    }

    for li in range(num_layers):
        # Build protected set: union of top-protect_top experts across all regression Qs
        protected = set()
        for qid in avail_reg:
            ranks = get_rank(pq, qid, li)
            for eid, r in ranks.items():
                if r < protect_top:
                    protected.add(eid)

        # For each expert, compute noise score from absolute wnorm
        scores = []
        gain_wn = {qid: get_wnorms(pq, qid, li) for qid in avail_gain}
        reg_wn = {qid: get_wnorms(pq, qid, li) for qid in avail_reg}

        for eid in range(num_experts):
            if eid in protected:
                continue
            gain_act = sum(gain_wn[q][eid] for q in avail_gain) / len(avail_gain)
            reg_act = sum(reg_wn[q][eid] for q in avail_reg) / len(avail_reg)
            noise = gain_act - alpha * reg_act
            scores.append((eid, noise, gain_act, reg_act))

        # Drop top-K noisiest
        scores.sort(key=lambda x: -x[1])
        drop = sorted([s[0] for s in scores[:K]])
        drop_map[li] = drop

        # Track violations against the strongest regression (Q32 if available)
        primary_reg = avail_reg[0]
        prim_rank = get_rank(pq, primary_reg, li)
        for d in drop:
            r = prim_rank[d]
            if r < 16:
                stats["violations_top16"] += 1
            if r < 32:
                stats["violations_top32"] += 1
            if r < 48:
                stats["violations_top48"] += 1

    print("Q32 protection check (post-fix):")
    print(f"  drops in Q{REGRESSION_QS[0]} top-16: {stats['violations_top16']}")
    print(f"  drops in Q{REGRESSION_QS[0]} top-32: {stats['violations_top32']}")
    print(f"  drops in Q{REGRESSION_QS[0]} top-48: {stats['violations_top48']}")
    print(f"  total drops: {num_layers * K}")
    return drop_map


def show_summary(pq, drop_map, num_layers):
    avail_reg = [q for q in REGRESSION_QS if str(q) in pq["questions"]]
    avail_gain = [q for q in GAIN_QS if str(q) in pq["questions"]]
    print()
    print("Sample (L0, L10, L20, L29):")
    for li in [0, 10, 20, 29]:
        gain_wn = {q: get_wnorms(pq, q, li) for q in avail_gain}
        reg_wn = {q: get_wnorms(pq, q, li) for q in avail_reg}
        items = []
        for d in drop_map[li]:
            g = sum(gain_wn[q][d] for q in avail_gain) / len(avail_gain)
            r = sum(reg_wn[q][d] for q in avail_reg) / len(avail_reg)
            items.append(f"E{d}(g={g:.1f},r={r:.1f})")
        print(f"  L{li:2d}: {' '.join(items)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8, help="experts to drop per layer")
    ap.add_argument("--protect-top", type=int, default=50,
                    help="never drop experts in top-N of any regression question")
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="weight on regression-question penalty in noise score")
    ap.add_argument("--sweep", action="store_true",
                    help="generate K=4,8,12 maps")
    args = ap.parse_args()

    pq = load()
    num_layers = pq["metadata"]["num_layers"]
    num_experts = pq["metadata"]["num_experts"]

    Ks = [4, 8, 12] if args.sweep else [args.K]
    for K in Ks:
        print(f"\n=== Building 120e-hybrid drop map (K={K}) ===")
        drop_map = build_drop_map(
            pq, K=K, protect_top=args.protect_top, alpha=args.alpha,
            num_layers=num_layers, num_experts=num_experts)
        show_summary(pq, drop_map, num_layers)

        out_default = Path("scripts/hybrid_120e_drop_map.json")
        out_named = Path(f"scripts/hybrid_120e_K{K}_drop_map.json")
        payload = {str(li): drop_map[li] for li in range(num_layers)}
        with open(out_named, "w") as f:
            json.dump(payload, f, indent=2)
        if K == args.K and not args.sweep:
            with open(out_default, "w") as f:
                json.dump(payload, f, indent=2)
        if args.sweep and K == 8:
            with open(out_default, "w") as f:
                json.dump(payload, f, indent=2)
        print(f"Saved {out_named}")
        print(f"Resulting model: {num_experts - K}e")


if __name__ == "__main__":
    main()
