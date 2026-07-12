#!/usr/bin/env python3
"""
Derive an expert-drop map from a competence map (the gate_competence_map.py format)
for Qwen3.6-35B-A3B. Ranks experts per layer by an importance score aggregated across
categories, drops the lowest `--drop-count` per layer, and writes the drop-map JSON that
expert_drop_qwen35b.py consumes: {"0":[ids],...,"39":[ids],"mtp":[ids]}.

Competence-map input (produced by the profiler, validated by gate_competence_map.py):
  {"metadata":{"num_layers":L,"num_experts":E},
   "categories":{<cat>:{"<li>":[{"id":e,"wnorm":..,"tc":..,"neuron_act":[..]?}, ...E], ...L}, ...}}

Importance score per (layer, expert), aggregated over categories with --agg:
  tc         routing frequency (times selected in top-k over the calib corpus)  [default, REAP-style]
  wnorm      per-expert output-contribution norm
  wnorm_tc   wnorm * tc  (magnitude x frequency)
Cross-category aggregation --agg: sum (default) | max | mean.
Lowest-importance experts per layer are dropped (least-used / least-contributing).

MTP head (--mtp-strategy):
  global  drop the bottom-`drop-count` by importance aggregated over ALL main layers  [default]
  layer0  reuse layer 0's drop set
  none    omit "mtp"  (expert_drop_qwen35b.py then defaults MTP to layer 0's set)

Usage:
  python make_drop_map.py --competence-map results/competence_qwen35b.json \
      --drop-count 72 --score tc --agg sum --mtp-strategy global \
      -o results/drop_map_184e.json
"""
import argparse, json
from collections import defaultdict


def score_of(expert, kind):
    if kind == "tc":
        return float(expert.get("tc", 0.0))
    if kind == "wnorm":
        return float(expert.get("wnorm", 0.0))
    if kind == "wnorm_tc":
        return float(expert.get("wnorm", 0.0)) * float(expert.get("tc", 0.0))
    raise ValueError(f"unknown score {kind}")


def aggregate(values, how):
    if not values:
        return 0.0
    if how == "sum":
        return sum(values)
    if how == "max":
        return max(values)
    if how == "mean":
        return sum(values) / len(values)
    raise ValueError(f"unknown agg {how}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competence-map", required=True)
    ap.add_argument("--drop-count", type=int, default=72)
    ap.add_argument("--score", choices=["tc", "wnorm", "wnorm_tc"], default="tc")
    ap.add_argument("--agg", choices=["sum", "max", "mean"], default="sum")
    ap.add_argument("--mtp-strategy", choices=["global", "layer0", "none"], default="global")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    with open(args.competence_map) as f:
        cm = json.load(f)
    L = int(cm["metadata"]["num_layers"])
    E = int(cm["metadata"]["num_experts"])
    cats = list(cm["categories"].keys())
    if not (0 < args.drop_count < E):
        raise SystemExit(f"drop-count {args.drop_count} must be in (0,{E})")

    # importance[li][eid] = agg over categories of the per-expert score
    per_cat = defaultdict(list)  # (li,eid) -> [scores]
    for cat in cats:
        layers = cm["categories"][cat]
        for li in range(L):
            row = layers[str(li)]
            assert len(row) == E, f"{cat} L{li}: {len(row)} experts != {E}"
            for e in row:
                per_cat[(li, int(e["id"]))].append(score_of(e, args.score))
    importance = defaultdict(dict)  # li -> {eid: score}
    global_imp = defaultdict(list)  # eid -> [per-layer scores]
    for (li, eid), vals in per_cat.items():
        s = aggregate(vals, args.agg)
        importance[li][eid] = s
        global_imp[eid].append(s)

    drop_map = {}
    for li in range(L):
        ranked = sorted(range(E), key=lambda e: importance[li].get(e, 0.0))  # ascending = least important first
        drop_map[str(li)] = sorted(ranked[:args.drop_count])
        assert len(drop_map[str(li)]) == args.drop_count

    if args.mtp_strategy == "global":
        g = {e: aggregate(global_imp.get(e, []), "sum") for e in range(E)}
        ranked = sorted(range(E), key=lambda e: g[e])
        drop_map["mtp"] = sorted(ranked[:args.drop_count])
    elif args.mtp_strategy == "layer0":
        drop_map["mtp"] = list(drop_map["0"])
    # "none" -> omit; dropper defaults MTP to layer 0

    with open(args.output, "w") as f:
        json.dump(drop_map, f)

    # provenance report
    keep = E - args.drop_count
    print(f"[make_drop_map] competence_map={args.competence_map}")
    print(f"[make_drop_map] L={L} E={E} score={args.score} agg={args.agg} categories={len(cats)}")
    print(f"[make_drop_map] drop {args.drop_count}/layer -> keep {keep} "
          f"({100*args.drop_count/E:.1f}% dropped)")
    print(f"[make_drop_map] mtp_strategy={args.mtp_strategy}"
          f"{' (drop '+str(len(drop_map['mtp']))+')' if 'mtp' in drop_map else ' (omitted -> layer0 default)'}")
    # overlap sanity: how stable is the drop set across layers?
    from collections import Counter
    c = Counter()
    for li in range(L):
        c.update(drop_map[str(li)])
    always = sum(1 for e in range(E) if c[e] == L)
    never = sum(1 for e in range(E) if c[e] == 0)
    print(f"[make_drop_map] experts dropped in ALL {L} layers: {always}; never dropped: {never} "
          f"(low 'always' => layer-specific specialization)")
    print(f"[make_drop_map] wrote {args.output}")


if __name__ == "__main__":
    main()
