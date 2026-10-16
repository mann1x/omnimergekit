#!/usr/bin/env python3
"""Merge new Tier-A categories from a Tier-A-only competence run into an existing map's data.

Used for v9-ml: map A's data (expert_neuron_v9_agentic_code.json: 5 generic + 7 targeted_swe
from a 43-session Tier-B replay) gains generic_multilingual + generic_ifeval measured in a
separate Tier-A-only run, so the expensive Tier-B replay is not repeated.

That is only valid if the two runs share a basis. Gate, before anything is written:
  1. metadata: model shape, tier_a_max_tokens, tier_a_thinking, wnorm_mode and chat
     template file must be identical;
  2. the categories BOTH files carry (the 5 generic_*) were recomputed by the new run;
     their per-layer expert rankings by wnorm must reproduce the base file's
     (Spearman >= --min-spearman in every layer, top-16 sets overlapping >= --min-top16).
     A run that cannot reproduce the categories we already have cannot be trusted for the
     ones we add.
Only categories absent from the base are copied; the base's own categories are never
overwritten. The new categories are inserted after the last generic_* category.
"""
import argparse
import json
import sys


def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    for pos, i in enumerate(order):
        r[i] = float(pos)
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def wn(cat, layer):
    return [e["wnorm"] for e in sorted(cat[layer], key=lambda e: e["id"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="existing map data (kept verbatim)")
    ap.add_argument("--add", required=True, help="Tier-A-only run carrying the new categories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-spearman", type=float, default=0.95)
    ap.add_argument("--min-top16", type=int, default=13)
    a = ap.parse_args()

    base = json.load(open(a.base))
    add = json.load(open(a.add))
    bm, am = base["metadata"], add["metadata"]
    for k in ("num_layers", "num_experts", "intermediate_size", "hidden_size",
              "tier_a_max_tokens", "tier_a_thinking", "wnorm_mode", "chat_template_file"):
        if bm.get(k) != am.get(k):
            sys.exit(f"BASIS MISMATCH {k}: base={bm.get(k)!r} add={am.get(k)!r}")

    shared = [c for c in add["categories"] if c in base["categories"]]
    new = [c for c in add["categories"] if c not in base["categories"]]
    if not shared:
        sys.exit("no shared categories: cannot verify the basis, refusing to merge")
    if not new:
        sys.exit("nothing to add")
    worst = (float("inf"), "")    # (rho, where); compared on rho only
    fails = []
    for c in shared:
        for layer in base["categories"][c]:
            b, n = wn(base["categories"][c], layer), wn(add["categories"][c], layer)
            rho = spearman(b, n)
            tb = set(sorted(range(len(b)), key=lambda i: -b[i])[:16])
            tn = set(sorted(range(len(n)), key=lambda i: -n[i])[:16])
            ov = len(tb & tn)
            if rho < worst[0]:
                worst = (rho, f"{c} L{layer} top16={ov}")
            if rho < a.min_spearman or ov < a.min_top16:
                fails.append(f"{c} L{layer} rho={rho:.4f} top16={ov}")
        print(f"[basis] {c}: reproduced", flush=True)
    print(f"[basis] worst layer: rho={worst[0]:.4f} ({worst[1]})")
    if fails:
        print("\n".join(fails[:20]))
        sys.exit(f"BASIS CHECK FAILED in {len(fails)} layer(s); not merging")

    order = list(base["categories"])
    last_generic = max(i for i, c in enumerate(order) if c.startswith("generic_"))
    merged_order = order[:last_generic + 1] + new + order[last_generic + 1:]
    cats = {c: (base["categories"][c] if c in base["categories"] else add["categories"][c])
            for c in merged_order}
    meta = dict(bm)
    meta["categories"] = merged_order
    meta["merged_tier_a"] = {"from": a.add, "categories": new,
                             "tier_a_extra": am.get("tier_a_extra"),
                             "basis_check": {"shared": shared, "worst_spearman": worst[0],
                                             "min_spearman": a.min_spearman,
                                             "min_top16": a.min_top16}}
    json.dump({"metadata": meta, "categories": cats}, open(a.out, "w"))
    print(f"[merge] wrote {a.out}: {len(merged_order)} categories, added {new}")


if __name__ == "__main__":
    main()
