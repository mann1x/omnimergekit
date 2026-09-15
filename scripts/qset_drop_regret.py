"""How much does it COST to use the wrong q-set's drop map?

Set-difference is the wrong endpoint. At a fixed drop_count the cut lands in the
middle of a rank distribution, so experts ranked 30th and 31st swap on any
perturbation -- that churn has a floor that never reaches zero no matter how many
traces are profiled. Counting it says "782 experts differ" and implies the map has
not converged, when the swapped experts may be near-ties whose exchange changes
nothing about the shipped model.

So measure the endpoint instead. Both maps score the same experts; the drop map is
just "discard the drop_count lowest". If I build the model using map Y's drop set
but map X is the truth, the competence I threw away is sum_X(scores over D_Y), vs
sum_X(scores over D_X) had I used the right one. The difference -- REGRET -- is
expressed as a fraction of the layer's total routed mass. It is zero when the two
maps disagree only about near-ties, and large when they disagree about which
experts actually matter.

TWO NORMALIZATIONS, because the obvious one is misleading. As a share of the
LAYER's total routed mass the number is always tiny -- the bottom-30 of 128 experts
carry only a few percent of the traffic, so any choice among them looks free (the
cross-language control, swapping rust's map for go's, reads just 0.227%). The
honest denominator is the mass you INTENDED to discard: "I meant to throw away this
much; using the wrong map I threw away X% more than that". That scale separates a
near-tie reshuffle from a real disagreement.

Reported alongside it is BOUNDARY FLATNESS: the score gap between the last-dropped
and first-kept expert, as a share of layer mass. A flat boundary is the mechanism
that makes churn large and regret small at the same time.
"""
import argparse
import json


def score_of(e, kind):
    if kind == "tc":
        return float(e.get("tc", 0.0))
    if kind == "wnorm":
        return float(e.get("wnorm", 0.0))
    if kind == "wnorm_tc":
        return float(e.get("wnorm", 0.0)) * float(e.get("tc", 0.0))
    raise ValueError(kind)


ap = argparse.ArgumentParser()
ap.add_argument("map_a")
ap.add_argument("map_b")
ap.add_argument("--score", default="tc", choices=["tc", "wnorm", "wnorm_tc"])
ap.add_argument("--only-cats", default="",
                help="comma-separated targeted category suffixes to restrict to. "
                     "A convergence curve must be read on ONE basis: the 3q point "
                     "only exists for the languages with >=6 PASS traces, so the "
                     "1q and 2q points have to be recomputed over those same "
                     "languages or the curve compares different language mixes.")
ap.add_argument("--drop-count", type=int, default=30)
ap.add_argument("--label", default="")
args = ap.parse_args()

A = json.load(open(args.map_a))["categories"]
B = json.load(open(args.map_b))["categories"]
cats = sorted(c for c in A if c.startswith("targeted_") and c in B)
if args.only_cats:
    want = {("targeted_" + x) if not x.startswith("targeted_") else x
            for x in args.only_cats.split(",") if x.strip()}
    missing = sorted(want - set(cats))
    if missing:
        raise SystemExit("FATAL: --only-cats named %s, absent from one of the maps" % missing)
    cats = [c for c in cats if c in want]

print("=== drop-set REGRET %s(score=%s, drop_count=%d) ===" % (
    (args.label + " ") if args.label else "", args.score, args.drop_count))
print("  regret = extra competence discarded by using the OTHER map's drop set,")
print("           as %% of the layer's total routed mass. 0%% = the disagreement is")
print("           entirely near-ties and costs nothing.")
print()
print("  %-26s %10s %10s %10s %10s %10s" % ("category", "regret%", "REL.regret%", "churn/ly", "dropmass%", "boundary%"))

allr, allw, allb, allrel = [], [], [], []
for c in cats:
    regs, gaps, churn, rels, shares = [], [], [], [], []
    for li in sorted(A[c], key=int):
        ea = {e["id"]: score_of(e, args.score) for e in A[c][li]}
        eb = {e["id"]: score_of(e, args.score) for e in B[c][li]}
        ids = sorted(set(ea) & set(eb))
        tot_a = sum(ea[i] for i in ids)
        tot_b = sum(eb[i] for i in ids)
        if tot_a <= 0 or tot_b <= 0:
            continue
        ra = sorted(ids, key=lambda i: (ea[i], i))
        rb = sorted(ids, key=lambda i: (eb[i], i))
        da, db = set(ra[:args.drop_count]), set(rb[:args.drop_count])
        # regret measured in EACH map's own units, then take the worse direction:
        # neither map is "the truth", so the honest number is the larger mistake.
        disc_a = sum(ea[i] for i in da)
        disc_b = sum(eb[i] for i in db)
        r_ab = (sum(ea[i] for i in db) - disc_a) / tot_a
        r_ba = (sum(eb[i] for i in da) - disc_b) / tot_b
        regs.append(max(r_ab, r_ba))
        if disc_a > 0 and disc_b > 0:
            rels.append(max((sum(ea[i] for i in db) - disc_a) / disc_a,
                            (sum(eb[i] for i in da) - disc_b) / disc_b))
        shares.append(disc_a / tot_a)
        churn.append(len(da ^ db) // 2)
        # boundary flatness in A: gap between last dropped and first kept
        if len(ra) > args.drop_count:
            gaps.append((ea[ra[args.drop_count]] - ea[ra[args.drop_count - 1]]) / tot_a)
    if not regs:
        continue
    mr = sum(regs) / len(regs)
    mrel = sum(rels) / len(rels) if rels else 0.0
    print("  %-26s %9.3f%% %10.2f%% %10.2f %9.2f%% %9.4f%%"
          % (c, 100 * mr, 100 * mrel, sum(churn) / len(churn),
             100 * (sum(shares) / len(shares)), 100 * (sum(gaps) / len(gaps) if gaps else 0)))
    allrel.append(mrel)
    allr.append(mr)
    allw.append(max(regs))
    allb.append(sum(gaps) / len(gaps) if gaps else 0)

print("  %-26s %9.3f%% %10.2f%% %10s %9s %9.4f%%"
      % ("MEAN", 100 * sum(allr) / len(allr), 100 * sum(allrel) / len(allrel), "", "",
         100 * sum(allb) / len(allb)))
print()
mean_r = 100 * sum(allrel) / len(allrel)
print("  (verdict reads REL.regret — %% of the mass the map meant to discard. Compare")
print("   against the cross-language control run on the same maps; that is the scale")
print("   of a genuinely different language, and the ceiling this can be worth.)")
if mean_r < 0.5:
    print("  VERDICT: the disagreement is NEAR-TIE CHURN. Using the other q-set's drop")
    print("           map discards %.3f%% more competence on average — below the noise" % mean_r)
    print("           floor of any downstream eval. More q-sets would move the expert")
    print("           IDs and not the model.")
elif mean_r < 2.0:
    print("  VERDICT: small but real (%.2f%%). Worth one more q-set to see if it shrinks." % mean_r)
else:
    print("  VERDICT: MATERIAL (%.2f%%). The maps disagree about experts that matter;" % mean_r)
    print("           more traces are required.")
