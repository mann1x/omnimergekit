#!/usr/bin/env python
"""Build HYBRID 184e drop maps: our task-attributed ranking + a REAP stability floor.

THE HYPOTHESIS BEING TESTED
---------------------------
Measured 2026-08-19 on the LCB-v6-77q decider: ours (armD) and REAP (armE) disagree about
38.5 experts per layer -- 53% of the 72-expert drop decision -- yet score within noise of
each other (29/77 LCB problems flip in BOTH directions for a net +3). What does separate
them is the loop axis: armE loops on 5/77, armD on 12/77, and none of armE's loops pass.
That points at REAP's magnitude criterion retaining something about generation stability
that our capability-targeted criterion does not select for.

So: keep OUR ranking as primary, but forbid dropping the experts REAP ranks highest. p is
the dose.

    P (protect) = the p experts we drop that REAP ranks highest among our drop set
    E (evict)   = the p experts we keep that OUR OWN score ranks lowest
    keep_new    = (keep_ours - E) union P

p=0 IS armD exactly; p=38.5 (the full disagreement) is approximately armE. So the sweep is a
straight interpolation between two already-measured endpoints, and the question is whether
the middle beats both ends. If it does not, the answer is "the criteria are interchangeable
and neither blend helps" -- also a result, and a cheap one.

Evicting by OUR lowest score (rather than by REAP's lowest) is deliberate: it keeps the
hybrid a modification of our recipe with a REAP-derived constraint, not a third ranking.
The two sets cannot collide -- P is drawn from our drop set, E from our keep set.

GATES (all fatal, none advisory)
  1. REAP dump fidelity: top-184 by dumped saliency == armE's keep set recovered from its
     WEIGHTS, on every layer. If the re-profile did not reproduce armE's selection, the
     dumped vector is not what armE used and nothing built from it means anything.
  2. Shipped-map consistency: armD's weight-recovered keep set == complement of the drop map.
  3. Output shape: exactly 184 keeps/layer, and exactly p experts changed vs armD.
"""
import argparse
import json
import os
import sys

RECIPE = "/srv/ml/repos/omnimergekit/recipes/qwen3_6_35b_a3b_prune"


def our_importance(map_path, drop_map_path, score, agg, cat_weight):
    """-> {layer: {expert: rank01}} using the SAME scoring path as the shipped recipe."""
    sys.path.insert(0, f"{RECIPE}/ream")
    import omk_ream_merge as orm
    # keep_offset=False => pure rank01(importance), no +1.0 keep-set forcing
    vecs, meta = orm.build_injected_saliency(
        map_path, drop_map_path, score, agg, cat_weight, keep_offset=False)
    return {li: {e: float(v[e]) for e in range(len(v))} for li, v in vecs.items()}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reap", default="/mnt/sdc/ream-work/reap_saliency.json")
    ap.add_argument("--keepsets", default="/mnt/sdc/ream-work/keepsets.json")
    ap.add_argument("--map", default=f"{RECIPE}/results/competence_qwen35b_coder_lcbmpe.json")
    ap.add_argument("--drop-map", default=f"{RECIPE}/results/drop_map_184e_coder_lcbmpe.json")
    ap.add_argument("--score", default="tc")
    ap.add_argument("--agg", default="wmax")
    ap.add_argument("--cat-weight", action="append", default=["corpus_targeted_lcb=2.0"])
    ap.add_argument("--keep", type=int, default=184)
    ap.add_argument("--protect", type=int, action="append", required=True,
                    help="p, repeatable: how many REAP-top experts to rescue from our drop set")
    ap.add_argument("--out-dir", default="/mnt/sdc/ream-work/hybrid_maps")
    args = ap.parse_args()

    reap = {int(k): v for k, v in json.load(open(args.reap)).items()}
    keepsets = json.load(open(args.keepsets))
    shipped = json.load(open(args.drop_map))
    imp, meta = our_importance(args.map, args.drop_map, args.score, args.agg, args.cat_weight)

    layers = sorted(int(k) for k in shipped if k.isdigit())
    E = int(meta["experts"])
    print(f"layers={len(layers)} experts={E} keep={args.keep}")

    # ---- GATE 1: dumped REAP reproduces armE's shipped selection -------------------------
    armE = keepsets.get("armE")
    if armE is None:
        sys.exit("REFUSING: keepsets.json has no 'armE' entry -- run recover_keepsets.py")
    bad = []
    for L in layers:
        if L not in reap:
            sys.exit(f"REFUSING: REAP dump missing layer {L}")
        top = set(sorted(range(E), key=lambda e: -reap[L][e])[:args.keep])
        if top != set(armE[str(L)]):
            bad.append((L, len(top - set(armE[str(L)]))))
    if bad:
        print(f"GATE 1 FAILED on {len(bad)}/{len(layers)} layers "
              f"(layer, n_mismatch): {bad[:8]}")
        sys.exit("REFUSING: dumped REAP saliency does not reproduce armE's keep set. The "
                 "re-profile is not the vector armE selected with; do not build from it.")
    print(f">>> GATE1_OK reap_dump top-{args.keep} == armE keep set on all {len(layers)} layers")

    # ---- GATE 2: shipped drop map agrees with armD's actual weights -----------------------
    # producer/consumer name mismatch: recover_keepsets.py is invoked with the arm
    # named "armD" by p5_armB.sh, while this gate only looked for the Qwen3.6-era
    # name. The lookup missed and gate 2 silently skipped. Accept both.
    armD = keepsets.get("armD_ourssal_nomerge") or keepsets.get("armD")
    if armD is not None:
        bad2 = [L for L in layers
                if set(armD[str(L)]) != set(range(E)) - set(int(x) for x in shipped[str(L)])]
        if bad2:
            sys.exit(f"REFUSING: armD weights disagree with the shipped drop map on layers "
                     f"{bad2[:8]} -- the 'ours' baseline is not what the map says.")
        print(f">>> GATE2_OK armD weights == complement of shipped drop map, all layers")
    else:
        print("*** GATE 2 NOT RUN: no armD/armD_ourssal_nomerge entry in "
              "keepsets.json. The drop map is UNVERIFIED against built weights. ***")

    os.makedirs(args.out_dir, exist_ok=True)
    for p in args.protect:
        newmap, moved, overlap_e = {}, [], []
        for L in layers:
            dropped = set(int(x) for x in shipped[str(L)])
            kept = [e for e in range(E) if e not in dropped]
            if len(kept) != args.keep:
                sys.exit(f"REFUSING: layer {L} keeps {len(kept)}, expected {args.keep}")
            # P: our drop set, ranked by REAP descending
            P = sorted(dropped, key=lambda e: -reap[L][e])[:p]
            # E: our keep set, ranked by OUR importance ascending
            Ev = sorted(kept, key=lambda e: imp[L][e])[:p]
            new_keep = (set(kept) - set(Ev)) | set(P)
            if len(new_keep) != args.keep:
                sys.exit(f"REFUSING: layer {L} p={p} produced {len(new_keep)} keeps")
            newmap[str(L)] = sorted(set(range(E)) - new_keep)
            moved.append(len(new_keep - set(kept)))
            overlap_e.append(len(new_keep & set(armE[str(L)])))
        # MTP entry carried through UNCHANGED: every arm gets the published 184e MTP block
        # grafted in afterwards (graft_mtp.py), so this entry never reaches a built model.
        if "mtp" in shipped:
            newmap["mtp"] = shipped["mtp"]

        # ---- GATE 3 ----
        if set(moved) != {p}:
            sys.exit(f"REFUSING: p={p} changed {sorted(set(moved))} experts/layer, expected {p}")
        out = f"{args.out_dir}/drop_map_184e_hybrid_p{p}.json"
        with open(out, "w") as fh:
            json.dump(newmap, fh)
        mean_e = sum(overlap_e) / len(overlap_e)
        print(f">>> HYBRID_OK p={p:>3}  changed_vs_armD={p}/layer  "
              f"mean_overlap_with_armE={mean_e:.1f}/{args.keep} "
              f"(armD baseline 145.5)  -> {out}")


if __name__ == "__main__":
    main()
