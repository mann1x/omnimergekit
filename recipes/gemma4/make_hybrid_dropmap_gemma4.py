#!/usr/bin/env python
"""Build HYBRID v7 drop maps: OUR task-attributed ranking + a REAP stability floor.

Port of the A3B `runhost/make_hybrid_dropmap.py` to Gemma-4 98e. Same construction:

    P (protect) = the p experts WE drop that REAP ranks highest among our drop set
    Ev (evict)  = the p experts WE keep that OUR OWN score ranks lowest
    keep_new    = (keep_ours - Ev) union P

p=0 is the shipped cut exactly. p is bounded above by the measured ours-vs-REAP
disagreement (Gemma-4: ~23 of 30 dropped/layer -- see the plan's P0.5 result).

WHAT IS DIFFERENT FROM A3B, AND WHY IT MATTERS
----------------------------------------------
The A3B builder picks Ev as "the p experts we keep that our own score ranks lowest", with
NO notion of a protected set. v7 has one: the **agentic_eog force-keep list** (46
layer:expert pairs) that STD16 was built around to fix generation looping. Those experts are
force-kept precisely BECAUSE our score ranks them low -- so a naive port would evict the loop
protection first. Ev therefore EXCLUDES the eog set, and a fatal gate asserts 0/46 evicted.

Our per-expert score is not re-derived by hand: it is computed with the SHIPPED producer's
own functions (generate_drop_map_v5fk.per_class_scores / normalize_per_class_per_layer /
aggregate) under the args recorded in the shipped map's .summary.json.
"""
import argparse
import json
import os
import statistics
import sys

import numpy as np



def load_force_keep(summary_path):
    sm = json.load(open(summary_path))
    fk = set()
    raw = sm["args"].get("force_keep") or ""
    for t in [x for x in raw.split(",") if x.strip()]:
        li, ei = t.split(":")
        fk.add((int(li), int(ei)))
    return fk, sm["args"]


def our_scores(prod, args, data_path):
    """[L, E] score using the SHIPPED producer's own scoring path."""
    cd = json.load(open(data_path))
    classes = args["classes"]
    weights = np.array([float(w) for w in args["class_weights"]], dtype=np.float64)
    s = prod.per_class_scores(cd, classes, float(args["alpha"]),
                              outlier_thresh=float(args["outlier_wnorm_thresh"]),
                              outlier_mode=args["outlier_mode"],
                              score_mode=args["score_mode"])
    s = prod.normalize_per_class_per_layer(s, mode=args["normalize"])
    agg = prod.aggregate(s, weights, args["strategy"])
    # breadth_bonus rewards experts scoring moderately across many classes. It is part of
    # the shipped score, not an option -- omitting it moves the bottom of the ranking, which
    # is exactly where Ev is drawn from.
    bb = float(args.get("breadth_bonus") or 0.0)
    if bb > 0:
        agg = agg + bb * s.mean(axis=0)
    return agg


def reproduce_shipped(prod, args, agg, floor_data, floor_map, fk):
    """Re-run the producer's OWN make_drop_map + force-keep to reproduce the shipped map.

    Only valid because the shipped args say protect_strategy='same' (=> protect_score None)
    and class_protect_floor=0 (=> s_norm None). Asserted, not assumed.
    """
    if args["protect_strategy"] != "same" or int(args["class_protect_floor"]) != 0:
        raise SystemExit("reproduce_shipped: shipped args use a protect path this port "
                         "does not implement (protect_strategy/class_protect_floor)")
    v4_pooled = None
    if floor_data:
        v4_pooled = prod.v4_pooled_score(json.load(open(floor_data)), float(args["alpha"]),
                                         outlier_thresh=float(args["outlier_wnorm_thresh"]),
                                         outlier_mode=args["outlier_mode"],
                                         score_mode=args["score_mode"])
    per_layer = None
    if floor_map:
        raw = json.load(open(floor_map))
        raw = raw.get("floor_per_layer", raw)
        per_layer = {int(k): int(v) for k, v in raw.items()}
    drop, _bt, _fa, _v4 = prod.make_drop_map(
        agg, int(args["target"]), int(args["protect_top"]),
        s_norm=None, class_protect_floor=0, protect_score=None,
        v4_pooled=v4_pooled, v4_floor_top=int(args["v4_floor_top"]),
        v4_floor_per_layer=per_layer)
    drop = {li: set(int(e) for e in v) for li, v in enumerate(drop)} \
        if isinstance(drop, list) else {int(k): set(int(e) for e in v) for k, v in drop.items()}
    E = agg.shape[1]
    n_drop = E - int(args["target"])
    for (li, e) in fk:                       # force-keep: pin, then rebalance the budget
        drop[li].discard(e)
    for li in drop:
        if len(drop[li]) < n_drop:
            keep_now = set(range(E)) - drop[li]
            fkl = {x for (l, x) in fk if l == li}
            cand = sorted((x for x in keep_now if x not in fkl), key=lambda x: agg[li][x])
            for x in cand[:n_drop - len(drop[li])]:
                drop[li].add(x)
    return drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reap", required=True)
    ap.add_argument("--reap-variant", default="published", choices=["published", "effective"])
    ap.add_argument("--drop-map", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--data", required=True, help="expert_neuron_v7_code.json")
    ap.add_argument("--producer-dir", required=True)
    ap.add_argument("--keep", type=int, default=98)
    ap.add_argument("--protect", type=int, action="append", required=True)
    ap.add_argument("--floor-data", default=None)
    ap.add_argument("--floor-map", default=None)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.producer_dir)
    import generate_drop_map_v5fk as prod

    fk, pargs = load_force_keep(a.summary)
    dm = {int(k): sorted(v) for k, v in json.load(open(a.drop_map)).items()}
    reap = json.load(open(a.reap))[a.reap_variant]
    E = json.load(open(a.reap))["meta"]["num_experts"]
    sc = our_scores(prod, pargs, a.data)
    print(f"[hybrid] layers={len(dm)} experts={E} keep={a.keep} "
          f"force_keep={len(fk)} reap_variant={a.reap_variant}")

    # GATE 1 -- shipped-map consistency, via the producer's OWN make_drop_map.
    # Ev is drawn from the BOTTOM of this ranking, which is exactly where an approximate
    # score path goes wrong -- so this must reproduce the shipped cut, not merely correlate.
    recon = reproduce_shipped(prod, pargs, sc, a.floor_data, a.floor_map, fk)
    ov, exact = [], 0
    for li, drop in dm.items():
        r = recon[li]
        ov.append(len(r & set(drop)) / len(set(drop)))
        exact += int(r == set(drop))
    mo = statistics.mean(ov)
    print(f"[gate1] shipped-map reproduction: mean drop-overlap {mo:.4f} "
          f"min {min(ov):.4f} | layers EXACT {exact}/{len(dm)}")
    if exact != len(dm):
        raise SystemExit(f"GATE1 FAIL: reproduced the shipped drop map exactly on only "
                         f"{exact}/{len(dm)} layers (mean overlap {mo:.4f}). Ev would be "
                         f"drawn from a ranking that is not the shipped one.")

    # GATE 2 -- eog experts must all be inside the shipped keep set to begin with.
    bad = [(l, e) for (l, e) in fk if e in set(dm[l])]
    if bad:
        raise SystemExit(f"GATE2 FAIL: shipped map already drops eog experts: {bad[:5]}")
    print(f"[gate2] all {len(fk)} eog experts present in shipped keep set")

    os.makedirs(a.out_dir, exist_ok=True)
    for p in a.protect:
        newmap, changed, evicted_eog = {}, [], 0
        for li, drop in dm.items():
            drop_s = set(drop)
            keep_ours = set(range(E)) - drop_s
            fkl = {e for (l, e) in fk if l == li}
            # P: of the experts WE drop, the p that REAP ranks highest
            P = set(sorted(drop_s, key=lambda e: -reap[str(li)][e])[:p])
            # Ev: of the experts WE keep, the p our own score ranks lowest -- eog EXCLUDED
            evictable = [e for e in keep_ours if e not in fkl]
            Ev = set(sorted(evictable, key=lambda e: sc[li][e])[:p])
            evicted_eog += len(Ev & fkl)
            keep_new = (keep_ours - Ev) | P
            if len(keep_new) != a.keep:
                raise SystemExit(f"GATE3 FAIL layer {li}: keep={len(keep_new)} != {a.keep}")
            ch = len(keep_new - keep_ours)
            if ch != p:
                raise SystemExit(f"GATE4 FAIL layer {li}: changed={ch} != p={p}")
            changed.append(ch)
            newmap[str(li)] = sorted(set(range(E)) - keep_new)
        if evicted_eog:
            raise SystemExit(f"GATE5 FAIL p={p}: {evicted_eog} eog experts evicted")
        outp = os.path.join(a.out_dir, f"v7_hybrid_p{p}_{a.reap_variant}_drop_map.json")
        json.dump(newmap, open(outp, "w"), indent=1)
        print(f"[hybrid] p={p:3d}  changed/layer={statistics.mean(changed):.2f}  "
              f"eog_evicted=0/{len(fk)}  -> {os.path.basename(outp)}")
    print(">>> HYBRID_MAPS_DONE")


if __name__ == "__main__":
    main()
