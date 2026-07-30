"""Print an OLD-vs-NEW comparison table from two (or more) summary_*.json files.

  python -m single_turn_seed_harness.tabulate results/summary_old-gmain.json \
                                              results/summary_new-g0709.json
"""
from __future__ import annotations

import json
import sys

_ROWS = [
    ("total generations", lambda s: s["total"]),
    ("fail-rate", lambda s: "%d/%d = %.1f%%" % (s["fails"], s["total"],
                                                100.0 * s["fail_rate"])),
    ("CLEAN", lambda s: s["verdicts"].get("CLEAN", 0)),
    ("RUNAWAY", lambda s: s["verdicts"].get("RUNAWAY", 0)),
    ("THINK_EXPLODE", lambda s: s["verdicts"].get("THINK_EXPLODE", 0)),
    ("CORRUPT", lambda s: s["verdicts"].get("CORRUPT", 0)),
    ("ABORT", lambda s: s["verdicts"].get("ABORT", 0)),
    ("len chars p50", lambda s: s["completion_len_chars"]["p50"]),
    ("len chars p90", lambda s: s["completion_len_chars"]["p90"]),
    ("tokens p50", lambda s: s["completion_tokens"]["p50"]),
    ("tokens p90", lambda s: s["completion_tokens"]["p90"]),
]


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        sys.exit("usage: tabulate summary_A.json [summary_B.json ...]")
    summaries = [json.load(open(p)) for p in argv]
    names = [s["name"] for s in summaries]
    w0 = max(len("metric"), *(len(r[0]) for r in _ROWS)) + 1
    wc = max(14, *(len(n) for n in names)) + 1
    header = "metric".ljust(w0) + "".join(n.rjust(wc) for n in names)
    print("\n==== single-turn seed sweep: OLD vs NEW ====")
    print(header)
    print("-" * len(header))
    for label, fn in _ROWS:
        row = label.ljust(w0)
        for s in summaries:
            row += str(fn(s)).rjust(wc)
        print(row)
    print("\n(fail = RUNAWAY + THINK_EXPLODE + CORRUPT; lower is better)")


if __name__ == "__main__":
    main()
