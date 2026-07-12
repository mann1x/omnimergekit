#!/usr/bin/env python3
"""T176 — finite-value GATE for a rebuilt competence map (base or targeted).

The whole point of T176 is that the legacy fp16 producer wrote finite-but-
pathological wnorm values (~1e18) with no guard, which silently corrupted the
drop-map floor. Before any map is allowed to drive a prune, it must pass this
gate. Exits non-zero on any hard failure so an orchestrator can stop the chain.

Checks per (category, layer, expert):
  - wnorm / tc / (neuron_act if present) are FINITE (no NaN/Inf)   [HARD]
  - |wnorm| <= --max-wnorm (default 1e6; the producer's own outlier ceiling) [HARD]
  - every category covers num_layers layers, each with num_experts experts [HARD]
Plus a non-degeneracy report for the multilingual category (the T176 addition):
  per-layer wnorm spread (min/median/max) — flags a layer whose top-vs-median
  ratio is ~1 (all experts equal → no signal → nothing to protect).            [WARN]
"""
import argparse
import json
import math
import sys


def all_finite(x):
    """Finite check that accepts scalars OR lists (neuron_act is a per-neuron array)."""
    if x is None:
        return True
    if isinstance(x, (list, tuple)):
        return all(all_finite(v) for v in x)
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--max-wnorm", type=float, default=1e6)
    ap.add_argument("--ml-category", default="generic_multilingual",
                    help="category name to report multilingual spread for "
                         "(tier_a_run names Tier-A cats generic_<key>).")
    args = ap.parse_args()

    with open(args.map) as f:
        d = json.load(f)
    meta = d["metadata"]
    L = int(meta["num_layers"])
    E = int(meta["num_experts"])
    cats = list(d["categories"].keys())
    print(f"[gate] map={args.map}")
    print(f"[gate] categories({len(cats)})={cats}")
    print(f"[gate] num_layers={L} num_experts={E}")

    hard_fail = 0
    n_nonfinite = 0
    n_over = 0
    over_examples = []
    for cat in cats:
        per_layer = d["categories"][cat]
        if len(per_layer) != L:
            print(f"[gate] HARD: category {cat} has {len(per_layer)} layers != {L}")
            hard_fail += 1
        for li in range(L):
            row = per_layer.get(str(li))
            if row is None or len(row) != E:
                print(f"[gate] HARD: {cat} L{li} has {0 if row is None else len(row)} experts != {E}")
                hard_fail += 1
                continue
            for e in row:
                w, tc = e.get("wnorm"), e.get("tc")
                na = e.get("neuron_act")
                if not all_finite(w) or not all_finite(tc) or not all_finite(na):
                    n_nonfinite += 1
                    if len(over_examples) < 6:
                        over_examples.append(f"{cat} L{li} e{e.get('id')} wnorm={w} tc={tc}")
                elif abs(float(w)) > args.max_wnorm:
                    n_over += 1
                    if len(over_examples) < 6:
                        over_examples.append(f"{cat} L{li} e{e.get('id')} |wnorm|={abs(float(w)):.3g}")

    if n_nonfinite:
        print(f"[gate] HARD: {n_nonfinite} non-finite wnorm/tc/neuron_act cells")
        hard_fail += 1
    if n_over:
        print(f"[gate] HARD: {n_over} cells with |wnorm| > {args.max_wnorm:g}")
        hard_fail += 1
    if over_examples:
        print("[gate] examples:")
        for x in over_examples:
            print(f"         {x}")

    # multilingual non-degeneracy report (WARN-only)
    mlcat = args.ml_category if args.ml_category in d["categories"] else None
    if mlcat is None:
        # fall back to any category containing 'multilingual'
        cand = [c for c in cats if "multilingual" in c]
        mlcat = cand[0] if cand else None
    if mlcat:
        print(f"[gate] multilingual category = {mlcat}: per-layer wnorm spread")
        flat_layers = 0
        for li in range(L):
            ws = sorted((float(e["wnorm"]) for e in d["categories"][mlcat][str(li)]), reverse=True)
            wmax, wmed, wmin = ws[0], ws[len(ws) // 2], ws[-1]
            ratio = wmax / wmed if wmed not in (0.0,) else float("inf")
            flag = ""
            if abs(ratio - 1.0) < 0.02:
                flag = "  <-- DEGENERATE (top≈median: no protectable signal)"
                flat_layers += 1
            if li < 3 or li >= L - 2 or flag:
                print(f"         L{li:>2}: max={wmax:9.3f} med={wmed:9.3f} min={wmin:9.3f} top/med={ratio:5.2f}{flag}")
        if flat_layers:
            print(f"[gate] WARN: {flat_layers} multilingual layers look degenerate")
    else:
        print("[gate] WARN: no multilingual category found in map")

    if hard_fail:
        print(f"[gate] RESULT: FAIL ({hard_fail} hard checks failed)")
        sys.exit(1)
    print("[gate] RESULT: PASS (all finite, within ceiling, full coverage)")


if __name__ == "__main__":
    main()
