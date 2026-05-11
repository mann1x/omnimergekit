#!/usr/bin/env python3
"""
Merge a rerun-of-unrecovered samples file back into a base samples file.

Pairs with `recover_failed_responses.py`:

    1. recover_failed_responses.py  → Tier 1 (extract from log), Tier 2 (list
       doc_ids that still need a fresh generation).
    2. lm_eval --predict_only on those doc_ids against a fixed server →
       produces a small `samples_<task>_rerun.jsonl`.
    3. merge_samples.py (this tool) → splices the rerun rows over the base
       rows (matched by doc_id), preserving the rest of the base file. Result
       is a full samples file ready for offline rescore.

Matching is by `doc_id` (with `idx` fallback). Each patch row replaces exactly
one base row; unmatched patch rows are reported on stderr and dropped.

Usage
-----
    python merge_samples.py \
        --base samples_gpqa_recovered.jsonl \
        --patch samples_gpqa_rerun.jsonl \
        --doc-ids unrecovered_doc_ids.txt \
        --output samples_gpqa_final.jsonl
        [--dry-run]
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def _doc_id(row: dict):
    """Return whatever key uniquely identifies a doc in lm-eval samples."""
    for k in ("doc_id", "idx"):
        if k in row:
            return row[k]
    return None


def _load_jsonl(p: Path) -> list[dict]:
    # \n-only split: avoid splitlines() — it breaks on \v \f \x1c which can
    # appear inside JSON-escaped model output and corrupt the parse.
    return [json.loads(line) for line in p.read_text().split("\n") if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path,
                    help="recovered samples_*.jsonl (output of recover_failed_responses.py)")
    ap.add_argument("--patch", required=True, type=Path,
                    help="rerun samples_*.jsonl with fresh generations for unrecovered doc_ids")
    ap.add_argument("--doc-ids", type=Path, default=None,
                    help="optional whitelist of doc_ids to apply from --patch "
                         "(one per line). When omitted, ALL patch rows are applied.")
    ap.add_argument("--output", required=True, type=Path,
                    help="merged samples_*.jsonl to write")
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts only; don't write output")
    args = ap.parse_args()

    for p in (args.base, args.patch):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    base = _load_jsonl(args.base)
    patch = _load_jsonl(args.patch)
    print(f"[+] base={len(base)} rows  patch={len(patch)} rows")

    allow: set | None = None
    if args.doc_ids and args.doc_ids.exists():
        allow_raw = [ln.strip() for ln in args.doc_ids.read_text().splitlines() if ln.strip()]
        # Doc_ids may be int — keep both string and int forms for permissive match.
        allow = set(allow_raw) | {int(x) for x in allow_raw if x.lstrip("-").isdigit()}
        print(f"[+] whitelist: {len(allow_raw)} doc_ids loaded from {args.doc_ids.name}")

    # Index base by doc_id (last occurrence wins — lm-eval can repeat on retry).
    base_index: dict = {}
    for i, row in enumerate(base):
        d = _doc_id(row)
        if d is not None:
            base_index[d] = i

    applied = 0
    unmatched: list = []
    skipped_filtered = 0
    for prow in patch:
        d = _doc_id(prow)
        if d is None:
            unmatched.append(("(no doc_id)", prow))
            continue
        if allow is not None and d not in allow and str(d) not in allow:
            skipped_filtered += 1
            continue
        if d in base_index:
            base[base_index[d]] = prow
            applied += 1
        else:
            unmatched.append((d, prow))

    print(f"[+] applied {applied} patch rows over base")
    if skipped_filtered:
        print(f"[+] skipped {skipped_filtered} patch rows (not in --doc-ids whitelist)")
    if unmatched:
        print(f"[!] {len(unmatched)} patch rows did not match any base doc_id; dropping:",
              file=sys.stderr)
        for d, _ in unmatched[:20]:
            print(f"      doc_id={d}", file=sys.stderr)
        if len(unmatched) > 20:
            print(f"      ... and {len(unmatched) - 20} more", file=sys.stderr)

    if args.dry_run:
        print("[+] dry-run; no output written")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for row in base:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[+] wrote {args.output} ({len(base)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
