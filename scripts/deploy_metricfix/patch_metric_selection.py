#!/usr/bin/env python3
"""Backport omk_eval's canonical-score selection fix (repo commit f06095a).

The old code's last resort was `next(iter(score_dict))` -- an arbitrary key.
lm-eval mixes bookkeeping entries into that dict, so when a template's declared
metric did not match what the task emitted, it returned "sample_len" (the row
COUNT) as the score: humaneval reported 164.0 for a run whose real pass@1 was
0.8963. A wrong plausible score is worse than a missing one; it lands in a table.

Applied surgically instead of syncing the whole file, because the run hosts carry
local work that a wholesale overwrite would destroy.

Idempotent: re-running is a no-op. Refuses unless the old block appears exactly
once, and ast.parse()s the result before writing.
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

MARKER = "LAST RESORT. Never hand back an arbitrary key"


def main() -> int:
    target, old_f, new_f = (Path(a) for a in sys.argv[1:4])
    src = target.read_text()
    old = old_f.read_text()
    new = new_f.read_text()

    if MARKER in src:
        print("already patched — no-op")
        return 0

    n = src.count(old)
    if n != 1:
        print(f"REFUSING: old block appears {n} times, expected exactly 1")
        return 2

    out = src.replace(old, new, 1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"REFUSING: result does not parse: {e}")
        return 3

    backup = target.with_suffix(".py.bak_metricfix")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"backed up -> {backup.name}")

    # Atomic swap: a bench launching mid-write must never see a partial file.
    tmp = target.with_suffix(".py.tmp_metricfix")
    tmp.write_text(out)
    tmp.replace(target)
    print("patched; AST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
