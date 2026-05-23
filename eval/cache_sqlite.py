"""Shared sqlite-backed response cache for the custom (non-lm-eval) omk_eval
backends (LCB, MultiPL-E).

Rationale (2026-05-23 directive): EVERY eval must resume through a sqlite DB.
The lm-eval backend already does (`--use_cache <sqlite>`), but the custom LCB
runner historically resumed from a plain `.jsonl` file and MultiPL-E had no
durable resume at all. This module is the single place that makes "all evals
use sqlite" true: both custom runners key their per-problem model responses
here, so a crash/OOM/timeout mid-run restarts from the last committed row
instead of from zero.

Backed by `sqlitedict` (the same library lm-eval uses), so the on-disk format
and the `out_dir/sqlite_cache/<prefix>_<model_tag>.db` convention are uniform
across the whole suite.

Keys are caller-defined strings:
  - LCB:        the problem `task_id`            (e.g. "lcb/leetcode/3566")
  - MultiPL-E:  f"{lang}::{problem_name}"        (e.g. "rs::HumanEval_0_has_close_elements")

Values are arbitrary JSON-serializable dicts (the response record). sqlitedict
pickles transparently; we keep values JSON-clean so a DB can also be dumped
with a one-liner for debugging.

Usage:
    from cache_sqlite import SqliteResponseCache
    with SqliteResponseCache(db_path) as cache:        # autocommit on writes
        if key in cache:
            rec = cache[key]
        else:
            rec = generate(...)
            cache[key] = rec        # committed immediately (crash-safe)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

try:
    from sqlitedict import SqliteDict
except ImportError as exc:  # pragma: no cover - dependency is via lm-eval[api]
    raise ImportError(
        "sqlitedict is required for the sqlite response cache "
        "(install via `pip install 'lm-eval[api]'` or `pip install sqlitedict`)."
    ) from exc


class SqliteResponseCache:
    """Thin, crash-safe key→record cache over a single sqlite file.

    `autocommit=True` flushes every write to disk immediately so a SIGKILL
    between problems never loses a completed generation — the same guarantee
    the old LCB runner got from `fsync` on its jsonl append, now durable in a
    real DB that the rest of the suite can query.
    """

    def __init__(self, db_path: str | Path, *, tablename: str = "responses",
                 flag: str = "c") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # flag="c": open for read/write, create if missing. journal_mode WAL
        # would speed concurrent writers, but the runners are single-writer;
        # the default rollback journal is the safest for crash recovery.
        self._db = SqliteDict(
            str(self.db_path), tablename=tablename, flag=flag, autocommit=True,
        )

    # ── mapping protocol ────────────────────────────────────────────────
    def __contains__(self, key: str) -> bool:
        return key in self._db

    def __getitem__(self, key: str) -> Any:
        return self._db[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._db[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._db.get(key, default)

    def keys(self) -> Iterator[str]:
        return self._db.keys()

    def __len__(self) -> int:
        return len(self._db)

    # ── lifecycle ───────────────────────────────────────────────────────
    def commit(self) -> None:
        self._db.commit()

    def close(self) -> None:
        try:
            self._db.commit()
        finally:
            self._db.close()

    def __enter__(self) -> "SqliteResponseCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def cache_db_path(out_dir: str | Path, prefix: str, model_tag: str) -> Path:
    """Canonical cache location, identical convention to the lm-eval backend's
    `out_dir/sqlite_cache/<prefix>_<model_tag>` (we append `.db` for clarity).
    Keeping the layout uniform means one rsync/cleanup rule covers every backend.
    """
    return Path(out_dir) / "sqlite_cache" / f"{prefix}_{model_tag}.db"
