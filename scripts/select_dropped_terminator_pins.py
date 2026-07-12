#!/usr/bin/env python3
"""select_dropped_terminator_pins.py — T202.4 targeted pin selector.

The top-K-per-layer ranker (rank_agentic_eog_pins.py) mostly re-pins experts the
prune already keeps. For the targeted force-keep TEST we want the opposite set:
the experts that are CURRENTLY DROPPED yet carry real terminator-prediction mass
on the 128e teacher — i.e. exactly the capacity the prune is throwing away.

Selection criterion (a dropped expert is pinned iff BOTH hold):
  * tc    >= --min-tc    (routed at >= this many of the emit positions)
  * wnorm >= --min-rms   (non-trivial emit-position RMS router weight)

Emits a `--force-keep "L:e,..."` string for generate_drop_map_v5.py. Each pin
evicts exactly one lowest-aggregate survivor, so the per-layer budget holds and
the rebuild is a clean single-variable delta vs the reproduced C6v3lcb map.

Usage:
  select_dropped_terminator_pins.py --eog-map <eog.json> --drop-map <drop.json> \
      [--min-tc 100] [--min-rms 0.25] [--metric wnorm] [--out pins.txt]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eog-map", required=True)
    ap.add_argument("--drop-map", required=True)
    ap.add_argument("--min-tc", type=int, default=100)
    ap.add_argument("--min-rms", type=float, default=0.25)
    ap.add_argument("--metric", default="wnorm")
    ap.add_argument("--category", default="agentic_eog")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cat = json.loads(Path(args.eog_map).read_text())["categories"][args.category]
    drop_raw = json.loads(Path(args.drop_map).read_text())
    dropped = {int(li): set(int(e) for e in v) for li, v in drop_raw.items()}

    pins = []  # (layer, expert, rms, tc)
    for li in sorted(int(x) for x in cat.keys()):
        dset = dropped.get(li, set())
        for r in cat[str(li)]:
            eid = int(r["id"])
            tc = int(r.get("tc", 0))
            rms = float(r.get(args.metric, 0.0))
            if eid in dset and tc >= args.min_tc and rms >= args.min_rms:
                pins.append((li, eid, rms, tc))

    print(f"=== targeted dropped-terminator pins "
          f"(tc>={args.min_tc}, {args.metric}>={args.min_rms}) ===")
    print(f"{'L':>3} {'expert':>6} {'emit-RMS':>9} {'tc':>5}")
    for li, eid, rms, tc in sorted(pins, key=lambda p: (-p[2], p[0])):
        print(f"{li:>3} {eid:>6} {rms:>9.2f} {tc:>5}")
    print(f"\nTOTAL pins (= evictions on rebuild): {len(pins)}")
    layers = sorted(set(p[0] for p in pins))
    print(f"layers touched: {layers}")

    fk = ",".join(f"{li}:{eid}" for li, eid, _, _ in sorted(pins))
    print(f"\n--force-keep {fk}")
    if args.out:
        Path(args.out).write_text(fk + "\n")
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
