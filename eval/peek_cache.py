"""Live peek at lm-eval SQLite cache while a benchmark runs.

Usage:
    peek_cache.py <cache_dir> [<task_glob>...]

Emits one line per new cached response (row added since last poll), with
row index, response length, a short preview, and a quality flag (empty,
fence-only, suspected-loop, ok). Designed for use under Monitor — pipe
to grep -E for the events you care about.

Example:
    peek_cache.py /srv/.../eval_results_smoke/cache humaneval_smoke20_chat mbpp_smoke40_chat
"""
from __future__ import annotations

import glob
import pickle
import sqlite3
import sys
import time
from pathlib import Path


def quality_flag(s: str) -> str:
    if not s or not s.strip():
        return "EMPTY"
    if s.strip().startswith("```") and s.strip().endswith("```") and len(s.strip()) < 30:
        return "FENCE_ONLY"
    # Loop detector: any 15-40 char tail repeated >=4 times
    tail = s[-200:]
    for cl in range(15, 40):
        if cl <= len(tail) and tail.count(tail[-cl:]) >= 4:
            return "LOOPING"
    if len(s) < 30:
        return "TINY"
    return "ok"


def peek_db(path: Path, seen: dict[str, set[str]]) -> int:
    new = 0
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
        cur = con.cursor()
        cur.execute("SELECT key, value FROM unnamed")
        rows = cur.fetchall()
        con.close()
    except sqlite3.OperationalError:
        return 0
    seen_set = seen.setdefault(path.name, set())
    for i, (k, v) in enumerate(rows):
        if k in seen_set:
            continue
        seen_set.add(k)
        try:
            val = pickle.loads(v)
        except Exception as e:
            print(f"[peek] {path.name} #{i+1:03d} DECODE_FAIL {e}", flush=True)
            continue
        if isinstance(val, list):
            val = val[0] if val else ""
        s = str(val).strip()
        flag = quality_flag(s)
        # Compress preview: first 80 chars, single-line, escape newlines
        preview = s[:120].replace("\n", "\\n")
        print(
            f"[peek] {path.name} #{i+1:03d} len={len(s):4d} {flag:10s} {preview}",
            flush=True,
        )
        new += 1
    return new


def main():
    if len(sys.argv) < 2:
        print("usage: peek_cache.py <cache_dir> [<task_glob>...]", file=sys.stderr)
        sys.exit(2)
    cache_dir = Path(sys.argv[1])
    globs = sys.argv[2:] or ["*"]
    seen: dict[str, set[str]] = {}
    while True:
        paths: list[Path] = []
        for g in globs:
            paths += [Path(p) for p in glob.glob(str(cache_dir / f"*{g}*.db"))]
        for p in sorted(set(paths)):
            peek_db(p, seen)
        time.sleep(15)


if __name__ == "__main__":
    main()
