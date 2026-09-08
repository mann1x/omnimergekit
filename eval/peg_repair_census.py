#!/usr/bin/env python3
"""Census (and optionally repair) broken generations sitting in an eval SQLite cache.

Motivation: a llama.cpp PEG parse failure used to throw -> HTTP 500 -> the
generation was DISCARDED. Cells measured on the pre-fix binary can therefore
carry rows that are empty or truncated where the model actually answered fine.
This tool finds those rows so they can be deleted and re-generated on the
patched binary, WITHOUT touching rows that are legitimately what the model said.

Report-only by default. --repair copies the db aside first, then deletes only
the keys this tool flagged, and prints the exact count it removed.

Two cache shapes are handled:
  * lm-eval CachingLM dbdict  -- one table, (key TEXT, value BLOB=pickle)
  * LCB SqliteResponseCache   -- table "responses", (key TEXT, value BLOB)
"""
from __future__ import annotations

import argparse
import json
import re
import pickle
import shutil
import sqlite3
import sys
from pathlib import Path

# Rows a bench is expected to hold when complete. A PEG-discarded generation
# usually never reaches the cache AT ALL -- the request errored -- so a missing
# row, not a short one, is the primary signature. Counting only broken rows
# cannot see an absence.
EXPECTED = {
    "arc_challenge_full": 1172,
    "gpqa_diamond_full": 198,
    "humaneval_full": 164,
    "humanevalplus_full": 164,
}

# NO blanket length floor. "The best answer is C" (20 chars) is a correct
# arc answer and "what are the two young boys holding?" (35) a correct ifeval
# one -- a 40-char floor flagged 366 such rows as broken. Truncation detection
# is bench-dependent; only defects that are defects on EVERY bench are flagged.


def expected_rows(cell: str) -> int | None:
    """Question count for a bench, from its name (aime_30 -> 30) or the table."""
    if cell in EXPECTED:
        return EXPECTED[cell]
    m = re.search(r"_(\d+)q?$", cell)
    return int(m.group(1)) if m else None


def _tables(con: sqlite3.Connection) -> list[str]:
    return [r[0] for r in con.execute(
        "select name from sqlite_master where type='table'")]


def _decode(blob: object) -> tuple[str | None, str]:
    """Return (text, how). text is None when the payload is not decodable."""
    if blob is None:
        return None, "null"
    if isinstance(blob, str):
        return blob, "text"
    if isinstance(blob, (bytes, bytearray)):
        try:
            obj = pickle.loads(blob)
            how = "pickle"
        except Exception:
            try:
                obj = json.loads(blob.decode("utf-8"))
                how = "json"
            except Exception:
                try:
                    return blob.decode("utf-8"), "raw-utf8"
                except UnicodeDecodeError:
                    return None, "undecodable"
        # lm-eval stores a list of responses; LCB stores a dict or a string.
        while isinstance(obj, (list, tuple)) and len(obj) == 1:
            obj = obj[0]
        if isinstance(obj, dict):
            # LCB SqliteResponseCache stores the generation under "completion";
            # "cleaned" is the post-extraction code and is legitimately empty
            # when extraction failed, so it must NOT stand in for the text.
            for k in ("completion", "text", "content", "response",
                      "generation", "output"):
                if k in obj and isinstance(obj[k], str):
                    return obj[k], how + "/" + k
            return None, how + "/dict-no-text-key"
        if isinstance(obj, str):
            return obj, how
        return str(obj), how + "/repr"
    return str(blob), "other"


def classify(text: str | None) -> str | None:
    """Return a defect label, or None when the row looks like a real answer."""
    if text is None:
        return "undecodable"
    if len(text) == 0:
        return "empty"
    if not text.strip():
        return "whitespace-only"
    # An opened-but-never-closed thinking channel means the tail was lost.
    if "<think>" in text and "</think>" not in text:
        return "unclosed-think"
    return None


def census(db: Path) -> tuple[dict[str, int], int, list[tuple[str, str]]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    counts: dict[str, int] = {}
    flagged: list[tuple[str, str]] = []
    total = 0
    for tbl in _tables(con):
        cols = [d[0] for d in con.execute(f"select * from {tbl} limit 1").description]
        if "key" not in cols or "value" not in cols:
            continue
        for key, value in con.execute(f"select key, value from {tbl}"):
            total += 1
            text, _ = _decode(value)
            label = classify(text)
            if label:
                counts[label] = counts.get(label, 0) + 1
                flagged.append((tbl, key))
    con.close()
    return counts, total, flagged


def repair(db: Path, flagged: list[tuple[str, str]]) -> int:
    if not flagged:
        return 0
    backup = db.with_suffix(db.suffix + ".prepegfix")
    if not backup.exists():
        shutil.copy2(db, backup)
        print(f"    backed up -> {backup.name}")
    con = sqlite3.connect(db)
    n = 0
    for tbl, key in flagged:
        n += con.execute(f"delete from {tbl} where key = ?", (key,)).rowcount
    con.commit()
    con.close()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="results dir, cell dir, or a single .db")
    ap.add_argument("--repair", action="store_true",
                    help="delete flagged rows (backs the db up first)")
    args = ap.parse_args()

    root = Path(args.root)
    dbs = [root] if root.is_file() else sorted(root.rglob("*.db"))
    if not dbs:
        print(f"no .db under {root}", file=sys.stderr)
        return 2

    grand: dict[str, int] = {}
    grand_total = grand_flagged = 0
    for db in dbs:
        counts, total, flagged = census(db)
        rel = db.relative_to(root) if not root.is_file() else db.name
        cell = str(rel).split("/")[0] if not root.is_file() else ""
        exp = expected_rows(cell)
        miss = ""
        if exp is not None and total < exp:
            miss = f"MISSING {exp - total}/{exp} "
            grand["missing-rows"] = grand.get("missing-rows", 0) + (exp - total)
        status = "clean" if not counts else ", ".join(
            f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  rows={total:<6} {miss}{status:<34} {rel}")
        for k, v in counts.items():
            grand[k] = grand.get(k, 0) + v
        grand_total += total
        grand_flagged += len(flagged)
        if args.repair and flagged:
            print(f"    deleted {repair(db, flagged)} rows")

    print(f"\nTOTAL rows={grand_total} flagged={grand_flagged} "
          f"({100.0 * grand_flagged / grand_total:.2f}%)" if grand_total else "\nno rows")
    for k, v in sorted(grand.items()):
        print(f"  {k:<20} {v}")
    if not args.repair and grand_flagged:
        print("\nreport-only. re-run with --repair to delete the flagged rows,")
        print("then re-launch that bench so they regenerate on the patched binary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
