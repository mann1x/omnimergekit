#!/usr/bin/env python3
"""Re-key a difficulty profile onto the mixed pool's own row ids.

WHY THIS EXISTS. The profiler samples PROBLEMS and keys its verdict by the problem id
it read out of the replay pool -- `mbpp/653`. The rebalanced mixed pool keys ROWS, and
a row is a problem *in a thinking mode*: `mbpp/653#N` and `mbpp/653#T` are different
rows. Handing the raw profile to --replay-difficulty therefore matches nothing at all,
and gepo_brevity refuses ("no row of this pool appears in the profile") -- which is the
right refusal, but it is not a fix.

WHY IT ONLY EVER ATTACHES TO ONE MODE. Unanimity is not a property of a problem. It is
a property of a problem AS SAMPLED: same problem, same G, different rendering, different
pass rate -- that is the entire reason both modes are in the pool. The profiler renders
no-think, so its verdict describes the no-think row and says nothing whatsoever about
the thinking row of the same problem. Copying it across would drop `#T` rows on
evidence gathered from a different distribution, and it would do so invisibly, because
a dropped row leaves no trace in the run log beyond a smaller count.

So: --profiled-think declares the mode the measurement was taken in, the mapping
attaches to rows of THAT mode only, and every other row stays unprofiled. Unprofiled
means KEPT (gepo_brevity.apply_difficulty), so the untouched mode loses nothing.
[[feedback_verify_eval_basis_by_hash_before_tabulating]]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="profile keyed by BASE problem id")
    ap.add_argument("--pool", required=True, help="the mixed pool being trained")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profiled-think", choices=["true", "false"], required=True,
                    help="the rendering the profile was MEASURED under")
    a = ap.parse_args()

    prof = json.loads(Path(a.profile).read_text())
    rows = [json.loads(x) for x in Path(a.pool).open() if x.strip()]
    want = a.profiled_think == "true"

    gs = {v.get("G") for v in prof.values()}
    if len(gs) != 1:
        sys.exit(f"REFUSE: profile mixes group sizes {sorted(gs)}; unanimity is a "
                 "property OF the group size and the entries are not comparable.")

    out, skipped_mode, unmatched = {}, 0, set(prof)
    for r in rows:
        base = r["id"].split("#")[0]
        if base not in prof:
            continue
        unmatched.discard(base)
        if bool(r["meta"].get("think")) != want:
            skipped_mode += 1
            continue
        out[r["id"]] = prof[base]

    if not out:
        sys.exit(f"REFUSE: no pool row matched the profile at think={a.profiled_think}. "
                 "Either the profile describes a different pool, or it was measured in "
                 "the mode this pool does not carry.")
    n_un = sum(1 for v in out.values() if v.get("unanimous"))
    print(f"mapped {len(out)} row(s) at think={a.profiled_think} "
          f"({n_un} unanimous, {len(out) - n_un} usable)")
    print(f"  left unprofiled on purpose: {skipped_mode} row(s) of the OTHER mode "
          "-- their pass rate was never measured")
    if unmatched:
        print(f"  WARNING: {len(unmatched)} profiled id(s) matched no pool row, "
              f"e.g. {sorted(unmatched)[:3]}")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
