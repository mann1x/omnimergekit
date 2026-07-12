#!/usr/bin/env python3
"""Evict length-capped LCB problems from a finished cell so a higher-cap rerun
re-generates ONLY them.

A length-capped problem is one whose cached response has
``finish_reason == "length"`` — the model hit ``--max-tokens`` and the code
block was sliced mid-generation, so ``exec()`` later raised a SyntaxError. That
is an artifact of the cap, not a model-quality failure, and is fixed by
re-running the offending problems at a larger ``max_gen_toks`` (the canonical
24576 retry).

The omk LCB runner keys its sqlite resume store by ``task_id`` ALONE (the key
does NOT include max_tokens — see eval/cache_sqlite.py). So a plain rerun at a
higher cap would just re-serve the truncated generation. This tool removes the
capped task_ids from BOTH:

  - the sqlite resume DB  (sqlite_cache/<prefix>_<tag>.db, table "responses")
  - the scoring artifact  (lcb_result.samples.jsonl)

After eviction, re-run omk_eval with a template whose ``gen.max_gen_toks`` is
24576 and whose ``name`` + ``cache.sqlite_prefix`` are unchanged: it resumes the
kept problems instantly and regenerates only the evicted ones, then rewrites
lcb_result.json + summary.json over the full 100.

Eval results are SACRED: this tool copies the db + samples to ``*.preretry.bak``
before mutating, and refuses to run if a backup already exists (unless --force).

Usage:
    lcb_evict_lengthcaps.py <cell_dir> [--inspect] [--force]
        <cell_dir>  dir containing sqlite_cache/ + lcb_result.samples.jsonl
        --inspect   list capped task_ids, change nothing
        --force     overwrite an existing *.preretry.bak
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

try:
    from sqlitedict import SqliteDict
except ImportError as exc:  # pragma: no cover
    print(f"ERR sqlitedict required: {exc}", file=sys.stderr)
    sys.exit(2)

BAK = ".preretry.bak"


def find_db(cell: Path) -> Path:
    dbs = sorted(glob.glob(str(cell / "sqlite_cache" / "*.db")))
    if not dbs:
        print(f"ERR no sqlite_cache/*.db under {cell}", file=sys.stderr)
        sys.exit(2)
    if len(dbs) > 1:
        print(f"ERR multiple .db files under {cell}: {dbs}", file=sys.stderr)
        sys.exit(2)
    return Path(dbs[0])


def capped_task_ids(db_path: Path) -> list[str]:
    db = SqliteDict(str(db_path), tablename="responses", flag="r")
    try:
        caps = []
        for k in db.keys():
            rec = db[k]
            if isinstance(rec, dict) and rec.get("finish_reason") == "length":
                caps.append(k)
        return sorted(caps)
    finally:
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_dir", type=Path)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cell = args.cell_dir
    if not cell.is_dir():
        print(f"ERR not a dir: {cell}", file=sys.stderr)
        return 2
    db_path = find_db(cell)
    samples = cell / "lcb_result.samples.jsonl"
    if not samples.exists():
        print(f"ERR no lcb_result.samples.jsonl under {cell}", file=sys.stderr)
        return 2

    caps = capped_task_ids(db_path)
    n_total = 0
    for _ in samples.open():
        n_total += 1
    print(f"cell={cell.name}  db={db_path.name}  samples_rows={n_total}  "
          f"capped(finish_reason=length)={len(caps)}")
    for t in caps:
        print(f"  - {t}")

    if not caps:
        print("nothing to evict.")
        return 0
    if args.inspect:
        print("\n(inspect mode — no changes written)")
        return 0

    db_bak = db_path.with_suffix(db_path.suffix + BAK)
    s_bak = samples.with_suffix(samples.suffix + BAK)
    for b in (db_bak, s_bak):
        if b.exists() and not args.force:
            print(f"ERR backup already exists: {b.name} (use --force to overwrite)",
                  file=sys.stderr)
            return 3
    shutil.copy2(db_path, db_bak)
    shutil.copy2(samples, s_bak)
    print(f"backed up -> {db_bak.name}, {s_bak.name}")

    # 1) evict from sqlite
    db = SqliteDict(str(db_path), tablename="responses", flag="w", autocommit=False)
    try:
        for t in caps:
            if t in db:
                del db[t]
        db.commit()
    finally:
        db.close()

    # 2) drop the capped rows from the scoring artifact
    capset = set(caps)
    kept = []
    for line in samples.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        tid = r.get("task_id") or r.get("doc_id")
        if tid not in capset:
            kept.append(line)
    with samples.open("w") as f:
        for line in kept:
            f.write(line + "\n")
    print(f"evicted {len(caps)} from sqlite; samples now {len(kept)} rows "
          f"(was {n_total}). Re-run omk_eval at max_gen_toks=24576 to refill.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
