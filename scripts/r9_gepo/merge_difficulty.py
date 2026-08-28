#!/usr/bin/env python3
"""Merge per-tier difficulty profiles into the single file the trainer reads.

Profiling runs per tier, because the tiers need different generation caps (mbpp ~1024,
gpqa 8192 -- a shared cap either mislabels the long tier or wastes hours on the short
one) and may run concurrently on different GPUs. Concurrent writers cannot share one
output file: each does a read-modify-write, so the second to finish would silently
discard the first. Hence one file per tier, merged here.

REFUSES on a G mismatch between inputs. Unanimity is a property OF the group size, so
profiles measured at different G describe different predicates and must not be pooled
into one file that the trainer will read as homogeneous.

Run:  python scripts/r9_gepo/merge_difficulty.py \\
          eval/replay/gepo_replay_pool.DIFFICULTY.*.json \\
          --out eval/replay/gepo_replay_pool.DIFFICULTY.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    merged: dict[str, dict] = {}
    gs: set[int] = set()
    for f in a.inputs:
        p = pathlib.Path(f)
        if not p.is_file():
            sys.exit(f"REFUSE: missing input {p}")
        d = json.loads(p.read_text())
        gs.update(v.get("G") for v in d.values())
        dup = set(d) & set(merged)
        if dup:
            sys.exit(f"REFUSE: {len(dup)} problem id(s) appear in more than one input "
                     f"(e.g. {sorted(dup)[:3]}). Two measurements of the same problem "
                     "must not be silently collapsed -- keep the one you trust.")
        merged.update(d)
        print(f"  + {p.name}: {len(d)} problems")

    if len(gs) > 1:
        sys.exit(f"REFUSE: inputs were measured at different G={sorted(gs)}. Unanimity "
                 "is a property OF the group size; these are not the same predicate.")

    tiers: dict[str, list[int]] = {}
    for v in merged.values():
        t = tiers.setdefault(v["kind"], [0, 0])
        t[0] += 1
        t[1] += int(v["unanimous"])
    pathlib.Path(a.out).write_text(json.dumps(merged, indent=2) + "\n")
    print(f"wrote {a.out}: {len(merged)} problems at G={gs.pop() if gs else '?'}")
    for k, (n, u) in sorted(tiers.items()):
        print(f"  {k}: {n} profiled, {u} unanimous, {n - u} usable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
