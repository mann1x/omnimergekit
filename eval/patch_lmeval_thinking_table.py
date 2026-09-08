#!/usr/bin/env python3
r"""omk-thinking-table: persist the THINKING channel into the lm-eval sqlite db.

WHY THIS EXISTS
  Fix-A makes `content` the graded text and uses `reasoning_content` only as a
  fallback when content is empty. That is correct for GRADING -- merging planning
  bullets into the answer contaminates rule-based scorers (IFEval no-comma,
  lowercase, ...). But Fix-A then DISCARDS the reasoning entirely, and the
  reasoning-sidecar (2026-05-29) only kept len(). So after a run finishes, the
  thinking text is gone forever.

  That is a real analysis hole. On v24/aime_100 (2026-07-26) 33/100 answers hit
  the 16,384-token generation wall, and the split between "the model ruminated
  until the budget was gone" and "the model wrote a long answer" was NOT
  decidable from disk -- reasoning_log.jsonl has char counts only, and the
  response cache stores just the content string. Diagnosing loop-vs-rumination
  required re-running the model.

WHAT IT DOES
  Adds a `thinking` table to the SAME sqlite file lm-eval already uses for
  `--use_cache`, so thinking is queryable next to the answers with one JOIN and
  no extra files to keep in sync:

      CREATE TABLE thinking (
          content_sha      TEXT,     -- sha256(content): the join key
          choice_idx       INTEGER,
          content_chars    INTEGER,
          reasoning_chars  INTEGER,
          reasoning        TEXT,     -- THE THINKING TEXT (what Fix-A dropped)
          content          TEXT,     -- the graded answer, for a self-contained row
          ts               REAL
      )

  lm-eval's own cache table (`unnamed`) is keyed by a hash of the REQUEST, which
  is not visible at this patch site (parse_generations sits below the cache
  hook). sha256(content) is derivable on both sides -- write it here, recompute
  it from the unpickled cache value later -- so it is the join key.

  Grading is untouched: the helper only observes. `tmp[...]` still receives
  exactly what Fix-A decided.

REQUIRES Fix-A (fix_a_lm_eval_patch.py) to be applied first -- it anchors on
Fix-A's `tmp[choices["index"]] = text`. Fails loud if that anchor is missing
rather than silently patching nothing.

Does NOT write reasoning_log.jsonl: the reasoning-sidecar patch may already be
installed and would double-write. The `thinking` table is a strict superset of
that jsonl; eval/query_thinking.py --emit-jsonl regenerates it for the older
consumers (scripts/analyze_track_results.py).

Usage: patch_lmeval_thinking_table.py /path/to/lm_eval/models/openai_completions.py
"""
import sys
from pathlib import Path

SENTINEL = "omk-thinking-table 2026-07-26"

# Fix-A's assignment inside LocalChatCompletion.parse_generations.
ANCHOR = '                    tmp[choices["index"]] = text'

CALLSITE = (
    '                    # omk-thinking-table 2026-07-26: observe both channels.\n'
    '                    # Grading is unchanged -- `text` above is still Fix-A\'s choice.\n'
    '                    _omkmsg = choices["message"]\n'
    '                    _omk_record_thinking(\n'
    '                        choices["index"],\n'
    '                        _omkmsg.get("content") or "",\n'
    '                        _omkmsg.get("reasoning_content")\n'
    '                        or _omkmsg.get("reasoning") or "")\n'
    '                    tmp[choices["index"]] = text'
)

HELPER = '''

# ---- omk-thinking-table 2026-07-26 -------------------------------------------
# Persist the thinking channel into the lm-eval --use_cache sqlite db, in a
# `thinking` table alongside the cached answers. Enabled only when
# OMK_THINKING_DB points at that db file; a no-op otherwise, so stock runs and
# non-omk callers are unaffected.
#
# NEVER raises: an analysis sidecar must not be able to fail an eval. Every
# failure path is swallowed after the first warning.
_OMK_THINKING_WARNED = False
_OMK_THINKING_DDL = (
    "CREATE TABLE IF NOT EXISTS thinking ("
    " content_sha TEXT, choice_idx INTEGER, content_chars INTEGER,"
    " reasoning_chars INTEGER, reasoning TEXT, content TEXT, ts REAL)"
)


def _omk_record_thinking(choice_idx, content, reasoning):
    """Append one (answer, thinking) row. Best-effort, never fatal."""
    global _OMK_THINKING_WARNED
    import os
    db = os.environ.get("OMK_THINKING_DB")
    if not db:
        return
    try:
        import hashlib
        import sqlite3
        import time
        sha = hashlib.sha256((content or "").encode("utf-8", "replace")).hexdigest()
        # busy_timeout: sqlitedict holds the same file and commits periodically,
        # so a writer collision is expected and must be waited out, not raced.
        cx = sqlite3.connect(db, timeout=60.0)
        try:
            cx.execute("PRAGMA busy_timeout=60000")
            cx.execute(_OMK_THINKING_DDL)
            cx.execute(
                "CREATE INDEX IF NOT EXISTS thinking_sha ON thinking(content_sha)")
            cx.execute(
                "INSERT INTO thinking VALUES (?,?,?,?,?,?,?)",
                (sha, choice_idx, len(content or ""), len(reasoning or ""),
                 reasoning or "", content or "", time.time()))
            cx.commit()
        finally:
            cx.close()
    except Exception as e:  # noqa: BLE001 - sidecar must never fail the eval
        if not _OMK_THINKING_WARNED:
            _OMK_THINKING_WARNED = True
            print(f"[omk-thinking-table] disabled after error: {e}", flush=True)
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    p = Path(sys.argv[1])
    src = p.read_text()
    if SENTINEL in src:
        print("skip-already-applied")
        return 0
    if ANCHOR not in src:
        print(f"FAIL: Fix-A anchor not found in {p}.\n"
              "      Apply fix_a_lm_eval_patch.py first (this patch stacks on it).",
              file=sys.stderr)
        return 1
    if src.count(ANCHOR) != 1:
        print(f"FAIL: Fix-A anchor appears {src.count(ANCHOR)}x in {p}; "
              "refusing to guess which one is the chat path.", file=sys.stderr)
        return 1
    p.write_text(src.replace(ANCHOR, CALLSITE, 1) + HELPER)
    print("ok-patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
