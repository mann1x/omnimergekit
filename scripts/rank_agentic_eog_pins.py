#!/usr/bin/env python3
"""rank_agentic_eog_pins.py — T202 ranker + decisive pre-check.

Reads a standalone `agentic_eog` competence map (built by
`expert_neuron_analysis_v5_targeted.py --variant agentic_eog`) and ranks, per
layer, the experts most responsible for PREDICTING the tool-turn terminator
`<|tool_response>`=50 (highest emit-position RMS wnorm). Emits a `--force-keep`
string for `generate_drop_map_v5.py` that pins those experts into the keep-set.

PRE-CHECK (the decisive, zero-GPU gate): diff the per-layer top-K terminator
experts against the CURRENT v7-coder dropped set. If the prune is already keeping
all of them, force-keep is a no-op and the RCA-as-stated is wrong → STOP before
spending GPU on a rebuild. If a substantial fraction of the terminator experts
are currently dropped, the prune is throwing away the stop-capability → proceed.

force-keep semantics (generate_drop_map_v5.py): pinning an already-kept expert is
a no-op; pinning a currently-dropped expert evicts exactly one lowest-aggregate
survivor (never another pin), preserving the per-layer budget. So we emit ALL
top-K per layer; the generator self-corrects.

Usage:
  rank_agentic_eog_pins.py \
      --eog-map  expert_neuron_v7_agentic_eog.json \
      --drop-map v7coder_g15f2440_drop_map.json \
      --topk 4 [--only-layers 24,25,26,27,28,29] [--min-tc 1] \
      --out v7coder_agentic_eog_forcekeep.txt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eog-map", required=True,
                    help="Standalone agentic_eog map JSON (category 'agentic_eog').")
    ap.add_argument("--drop-map", required=True,
                    help="Current v7-coder drop-map JSON {layer: [dropped expert ids]}.")
    ap.add_argument("--topk", type=int, default=4,
                    help="Top-K terminator experts to pin per layer (default 4).")
    ap.add_argument("--metric", default="wnorm",
                    help="Per-expert field to rank by (default wnorm = emit-position RMS).")
    ap.add_argument("--min-tc", type=int, default=1,
                    help="Require >= this many emit tokens routed to the expert before it "
                         "is eligible (default 1 — exclude experts never routed at emit "
                         "positions, whose wnorm is a meaningless 0).")
    ap.add_argument("--only-layers", default=None,
                    help="Comma-separated layer indices to restrict pinning to (e.g. "
                         "'24,25,26,27,28,29'). Default: all layers.")
    ap.add_argument("--category", default="agentic_eog")
    ap.add_argument("--out", default=None,
                    help="Write the --force-keep string here (also printed).")
    args = ap.parse_args()

    eog = json.loads(Path(args.eog_map).read_text())
    cats = eog["categories"]
    if args.category not in cats:
        raise SystemExit(f"category {args.category!r} not in map; have {list(cats)}")
    cat = cats[args.category]

    drop_raw = json.loads(Path(args.drop_map).read_text())
    dropped = {int(li): set(int(e) for e in v) for li, v in drop_raw.items()}

    layers = sorted(int(li) for li in cat.keys())
    if args.only_layers:
        want = set(int(x) for x in args.only_layers.split(","))
        layers = [li for li in layers if li in want]

    pins = {}                # layer -> [expert ids]
    total_pins = 0
    total_dropped_pins = 0   # == evictions the generator will perform
    layers_with_overlap = 0
    routed_per_layer = []
    print(f"=== agentic_eog pin ranking (top-{args.topk} by {args.metric}, "
          f"min_tc={args.min_tc}) ===")
    print(f"{'L':>3} {'routed':>6} {'topK experts (emit-RMS)':<40} "
          f"{'#currently_dropped':>18}")
    for li in layers:
        rows = cat[str(li)]
        elig = [r for r in rows if int(r.get("tc", 0)) >= args.min_tc]
        routed = len(elig)
        routed_per_layer.append(routed)
        elig.sort(key=lambda r: float(r.get(args.metric, 0.0)), reverse=True)
        top = elig[: args.topk]
        top_ids = [int(r["id"]) for r in top]
        pins[li] = top_ids
        total_pins += len(top_ids)
        dropped_here = [e for e in top_ids if e in dropped.get(li, set())]
        total_dropped_pins += len(dropped_here)
        if dropped_here:
            layers_with_overlap += 1
        shown = ", ".join(f"{int(r['id'])}({float(r.get(args.metric,0.0)):.2f})"
                          for r in top)
        print(f"{li:>3} {routed:>6} {shown:<40} {len(dropped_here):>18}")

    # ── PRE-CHECK verdict ────────────────────────────────────────────────────
    frac = (total_dropped_pins / total_pins) if total_pins else 0.0
    print("\n=== PRE-CHECK (decisive, zero-GPU) ===")
    print(f"  layers considered      : {len(layers)}")
    print(f"  total top-K pins        : {total_pins}")
    print(f"  pins currently DROPPED  : {total_dropped_pins}  "
          f"({frac*100:.0f}% of pins; = evictions on rebuild)")
    print(f"  layers with >=1 dropped : {layers_with_overlap}/{len(layers)}")
    print(f"  mean experts routed at emit/layer: "
          f"{sum(routed_per_layer)/max(1,len(routed_per_layer)):.1f}")
    if total_dropped_pins == 0:
        verdict = ("STOP — every top terminator expert is ALREADY kept by the prune. "
                   "Force-keep is a no-op; the RCA-as-stated does not hold. Do NOT rebuild.")
    elif frac < 0.10 and layers_with_overlap <= max(1, len(layers) // 10):
        verdict = ("MARGINAL — very little overlap; the loop is likely not a dropped-expert "
                   "problem. Reconsider before spending GPU.")
    else:
        verdict = ("PROCEED — a substantial share of terminator experts are being dropped; "
                   "protecting them should change the keep-set. Rebuild + gate.")
    print(f"\n  VERDICT: {verdict}")

    # ── force-keep string (all top-K; generator no-ops already-kept) ─────────
    fk = ",".join(f"{li}:{e}" for li in layers for e in pins[li])
    print(f"\n--force-keep {fk}")
    if args.out:
        Path(args.out).write_text(fk + "\n")
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
