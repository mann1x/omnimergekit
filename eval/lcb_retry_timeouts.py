#!/usr/bin/env python3
"""Retry LCB samples where the original run hit a client-side ReadTimeout.

Detects timeout rows by the fingerprint `chars=0 + reason starts with
"timeout"|"ReadTimeout"|"gen-error"` and `gen_secs >= 599` and rewrites
the samples.jsonl with those rows removed. Subsequent `lcb_llama_server.py`
runs in resume mode then re-generate ONLY those problems against the
already-running server (assuming a longer --http-timeout is now used).

Usage:
    lcb_retry_timeouts.py <samples.jsonl> [--inspect]
        --inspect   list what would be retried without rewriting

Example end-to-end retry on the 128e_nvfp4a16 LCB-55 run:
    SAMPLES=eval_results_vllm4bit/lcb_med_55q/128e_nvfp4a16/128e_nvfp4a16_lcb_med_55q.samples.jsonl

    # 1. List which task_ids will retry
    python eval/lcb_retry_timeouts.py "$SAMPLES" --inspect

    # 2. Strip them from the cache
    python eval/lcb_retry_timeouts.py "$SAMPLES"

    # 3. Restart server (or reuse existing) and re-run with 1200s timeout
    python eval/lcb/lcb_llama_server.py \\
        --name 128e_nvfp4a16 --base-url http://localhost:8195 \\
        --max-tokens 16384 --http-timeout 1200 \\
        --limit 999 --difficulty medium --min-date 2024-10-01 \\
        --output ${SAMPLES%.samples.jsonl}.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def is_timeout(row: dict) -> bool:
    """Heuristic: chars==0 AND (reason mentions timeout OR gen_secs >= 599)."""
    if row.get("passed"):
        return False
    if (row.get("completion_chars") or len(row.get("completion", "") or "")) > 0:
        return False
    reason = (row.get("reason") or "").lower()
    if "timeout" in reason or "readtimeout" in reason or "gen-error" in reason:
        return True
    gs = row.get("gen_secs") or 0.0
    return gs >= 599.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=Path)
    ap.add_argument("--inspect", action="store_true",
                    help="list timeout rows without modifying the file")
    ap.add_argument("--backup", default=".timeouts.bak",
                    help="suffix for the pre-rewrite backup (default: .timeouts.bak)")
    args = ap.parse_args()

    if not args.samples.exists():
        print(f"ERR no such file: {args.samples}", file=sys.stderr)
        return 2

    rows = []
    timeouts: list[dict] = []
    for line in args.samples.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.append(r)
        if is_timeout(r):
            timeouts.append(r)

    if not timeouts:
        print(f"No timeout rows found in {args.samples.name} "
              f"({len(rows)} total, {sum(1 for r in rows if r.get('passed'))} PASS)")
        return 0

    print(f"Found {len(timeouts)} timeout row(s) in "
          f"{args.samples.name} ({len(rows)} total):")
    for r in timeouts:
        print(f"  - {r.get('task_id'):24s}  "
              f"gen_secs={r.get('gen_secs', 0):.1f}  "
              f"reason={(r.get('reason') or '')[:70]}")

    if args.inspect:
        print("\n(inspect mode — no changes written)")
        return 0

    backup = args.samples.with_suffix(args.samples.suffix + args.backup)
    print(f"\nBacking up to {backup.name}")
    shutil.copy2(args.samples, backup)

    kept = [r for r in rows if not is_timeout(r)]
    with args.samples.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"Rewrote {args.samples.name}: kept {len(kept)} rows, "
          f"dropped {len(timeouts)} timeouts.")
    print("Next: re-run the LCB shim with a longer --http-timeout (e.g. 1200).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
