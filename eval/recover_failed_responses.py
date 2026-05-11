#!/usr/bin/env python3
"""
Recover 500-failed responses from an lm-eval run against llama-server.

Background
----------
llama-server's `--jinja` peg-gemma4 chat parser fails to serialize responses
that contain interleaved `<|channel>` tokens (model emits a channel switch
mid-response after CoT). The result is HTTP 500 returned to lm-eval, even
though the generation completed cleanly. The full model text is preserved
in the 500 error body's `"message"` field.

This tool walks the lm-eval RUNLOG (tee output of `lm_eval ... | tee RUNLOG`)
and the `samples_*.jsonl` produced by lm-eval, then:

  Tier 1 — extracts model text from each captured `code":500` payload in
           RUNLOG, patches the corresponding sample row (by sequential
           order — matches the Kth empty/failed sample to the Kth 500).
  Tier 2 — for samples that have no extractable text (e.g. 500 with no
           message body, or count mismatch), prints the doc_id list so
           those questions can be re-run separately.
  Tier 3 — emits a merged samples file ready for lm-eval rescoring
           (`lm_eval ... --predict_only` / offline rescore).

Usage
-----
    python recover_failed_responses.py \
        --samples eval_results_v4/.../samples_gpqa_*.jsonl \
        --runlog  eval_results_v4/.../eval_gpqa_*_run.log \
        --output  /tmp/samples_gpqa_recovered.jsonl \
        [--dry-run]

Caveats
-------
- Assumes `num_concurrent=1` in the lm-eval invocation. With concurrency,
  the K-th 500 doesn't necessarily map to the K-th failed sample.
- Assumes one 500 per failed sample. If lm-eval retried and got more 500s
  for the same question, only the last one in RUNLOG wins (matches lm-eval's
  behavior: it returns the last response it saw).
- The extracted "resps" is the full model text including the `<|channel>`
  token. Downstream scorers (lm-eval flex-extract) should still find the
  answer letter; verify after rescoring.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

# Pattern for the embedded 500 body in RUNLOG. lm-eval logs the warning as:
#   WARNING [models.api_models:479] API request failed with error message:
#   {"error":{"code":500,"message":"<full body>","type":"server_error"}}
# Or (older versions):
#   {"error":{"code":500,"message":"<full body>"}}
#
# The message itself contains escaped JSON. We extract on the FIRST 500 line.
ERR_500_PATTERN = re.compile(
    r'\{"error":\{"code":500,"message":"(?P<body>.*?)","type":"server_error"\}\}'
)
# Fallback without type field
ERR_500_PATTERN_NO_TYPE = re.compile(
    r'\{"error":\{"code":500,"message":"(?P<body>.*?)"\}\}'
)


def _unescape(s: str) -> str:
    """JSON-unescape the captured body string."""
    # Simulate JSON string decoding without wrapping in another JSON parse,
    # since the body may contain stray unescaped quotes from llama-server.
    return (
        s.replace("\\n", "\n")
         .replace("\\t", "\t")
         .replace('\\"', '"')
         .replace("\\\\", "\\")
    )


def extract_500_bodies(runlog: Path) -> list[str]:
    """Return list of model-text payloads, one per 500 found in RUNLOG."""
    text = runlog.read_text(encoding="utf-8", errors="replace")
    out: list[str] = []
    for m in ERR_500_PATTERN.finditer(text):
        out.append(_unescape(m.group("body")))
    if not out:
        for m in ERR_500_PATTERN_NO_TYPE.finditer(text):
            out.append(_unescape(m.group("body")))
    return out


_INVALID_MARKERS = {"[invalid]", "[INVALID]"}


def _flatten_strs(v) -> list[str]:
    flat: list[str] = []
    if isinstance(v, list):
        for x in v:
            if isinstance(x, list):
                flat.extend(s for s in x if isinstance(s, str))
            elif isinstance(x, str):
                flat.append(x)
    return flat


def is_empty_or_failed(row: dict) -> bool:
    """Row is "failed" if the raw `resps` is empty/whitespace.

    Note: we deliberately do NOT trust `filtered_resps` as evidence of a real
    response — lm-eval writes `["[invalid]"]` there when extraction fell back,
    which the original samples-row inspector would otherwise count as valid.
    Authority of recovery is the *raw* generation in `resps`.
    """
    if "resps" in row:
        flat = _flatten_strs(row["resps"])
        if not flat:
            return True
        # any non-whitespace, non-marker raw string = real generation
        for s in flat:
            ss = s.strip()
            if ss and ss not in _INVALID_MARKERS:
                return False
        return True
    # Fallback: no resps field at all → trust filtered_resps as last resort.
    flat = _flatten_strs(row.get("filtered_resps"))
    for s in flat:
        ss = s.strip()
        if ss and ss not in _INVALID_MARKERS:
            return False
    return True


def patch_row_with_text(row: dict, text: str) -> None:
    """Inject `text` into a sample row's resps slot."""
    # lm-eval task-agnostic shape: resps is list[list[str]] with one inner
    # list per repetition. For zeroshot/greedy we have 1×1.
    if isinstance(row.get("resps"), list):
        if row["resps"] and isinstance(row["resps"][0], list):
            row["resps"][0] = [text]
        else:
            row["resps"] = [[text]]
    else:
        row["resps"] = [[text]]
    # filtered_resps gets a string per filter; safest to clear so re-extraction runs
    if "filtered_resps" in row:
        row["filtered_resps"] = [text]
    row.setdefault("_recovery", {})["source"] = "RUNLOG-500-recovery"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", required=True, type=Path,
                   help="lm-eval samples_*.jsonl produced by the failed run")
    p.add_argument("--runlog", required=True, type=Path,
                   help="lm-eval RUNLOG (tee output) with the 500 payloads")
    p.add_argument("--output", required=True, type=Path,
                   help="merged/patched samples_*.jsonl to write")
    p.add_argument("--dry-run", action="store_true",
                   help="report counts only; don't write output")
    p.add_argument("--unrecovered-out", type=Path, default=None,
                   help="write unrecovered doc_id list here (one per line); "
                        "feed to a re-run via --doc-id-filter / a task subset.")
    p.add_argument("--task", default=None,
                   help="lm-eval task name (e.g. gpqa_diamond_cot_zeroshot). "
                        "If set, print a ready-to-run rerun + rescore command.")
    p.add_argument("--model-name", default=None,
                   help="model tag for the rerun cache prefix")
    args = p.parse_args()

    if not args.samples.exists():
        print(f"ERROR: samples file not found: {args.samples}", file=sys.stderr)
        return 1
    if not args.runlog.exists():
        print(f"ERROR: RUNLOG not found: {args.runlog}", file=sys.stderr)
        return 1

    # Load
    # Use \n-only split: splitlines() also breaks on \v \f \x1c which can
    # appear inside JSON-escaped model output and corrupt the parse.
    samples = [json.loads(line) for line in args.samples.read_text().split("\n") if line.strip()]
    print(f"[+] {args.samples.name}: {len(samples)} sample rows")

    # Identify empties, grouped by doc_id (lm-eval emits ONE row per filter,
    # so a single failed generation typically appears in 2+ rows with the
    # same doc_id — they should all receive the SAME recovered text).
    empty_idx = [i for i, r in enumerate(samples) if is_empty_or_failed(r)]
    print(f"[+] empty/failed rows: {len(empty_idx)}")

    # Order of unique failed doc_ids (preserves first-occurrence order, which
    # under num_concurrent=1 matches the RUNLOG event order).
    seen: set = set()
    unique_failed_docs: list = []
    rows_by_doc: dict = {}
    for i in empty_idx:
        d = samples[i].get("doc_id", samples[i].get("idx"))
        if d not in seen:
            seen.add(d)
            unique_failed_docs.append(d)
        rows_by_doc.setdefault(d, []).append(i)
    print(f"[+] unique failed doc_ids: {len(unique_failed_docs)} "
          f"({unique_failed_docs[:10]}{'...' if len(unique_failed_docs)>10 else ''})")

    # Extract 500 bodies from RUNLOG.
    bodies = extract_500_bodies(args.runlog)
    print(f"[+] {args.runlog.name}: {len(bodies)} captured 500 payloads")

    # Pair bodies → unique failed docs (NOT to rows directly). When there are
    # MORE bodies than unique failures (retries hit before lm-eval gave up),
    # the LAST body wins for that doc (mirrors lm-eval's behavior — final
    # observed response is the one it stored).
    #
    # Naive: with N bodies and K unique docs and N >= K, assign each doc the
    # body at position `(d_idx + 1) * N // K - 1` (the last body in its slice).
    # Better when retry counts per doc aren't known: take the LAST K bodies.
    # We take the last-K approach because lm-eval retries with backoff and the
    # final attempt per doc is what's most likely to make it into RUNLOG.
    K = len(unique_failed_docs)
    if len(bodies) >= K:
        chosen_bodies = bodies[-K:]
    else:
        # Fewer bodies than failed docs — best-effort, leave tail unpatched.
        chosen_bodies = bodies + [""] * (K - len(bodies))
    n_pair = min(K, len([b for b in chosen_bodies if b.strip()]))
    print(f"[+] Tier 1 — pairing {n_pair} bodies → {K} unique failed docs "
          f"(replicated across 2+ filter rows per doc)")

    patched_rows = 0
    patched_docs = 0
    for d, body in zip(unique_failed_docs, chosen_bodies):
        if not body.strip():
            continue
        for row_i in rows_by_doc[d]:
            patch_row_with_text(samples[row_i], body)
            patched_rows += 1
        patched_docs += 1
    print(f"[+] patched {patched_rows} rows ({patched_docs} unique docs) in-memory")

    # Tier 2 — list doc_ids that remain unrecovered
    still_empty = [
        i for i, r in enumerate(samples) if is_empty_or_failed(r)
    ]
    unrecovered_doc_ids: list = []
    if still_empty:
        print(f"[!] Tier 2 — {len(still_empty)} rows STILL EMPTY after recovery; "
              "doc_ids:", file=sys.stderr)
        for i in still_empty:
            r = samples[i]
            did = r.get("doc_id", r.get("idx", f"row-{i}"))
            unrecovered_doc_ids.append(did)
            print(f"      doc_id={did} target={r.get('target', '?')}", file=sys.stderr)

        if args.unrecovered_out and not args.dry_run:
            args.unrecovered_out.parent.mkdir(parents=True, exist_ok=True)
            args.unrecovered_out.write_text(
                "\n".join(str(d) for d in unrecovered_doc_ids) + "\n"
            )
            print(f"[+] wrote unrecovered doc_id list → {args.unrecovered_out}",
                  file=sys.stderr)

        # Print a ready-to-run rerun + rescore recipe.
        if args.task:
            tag = args.model_name or "RECOVERED"
            print("", file=sys.stderr)
            print("[i] Tier 2/3 — rerun + rescore recipe:", file=sys.stderr)
            print("    # 1. Re-run ONLY the unrecovered doc_ids with a fixed server", file=sys.stderr)
            print("    #    (e.g. llama-server WITHOUT --jinja, or with a patched parser).", file=sys.stderr)
            print("    #    Pass --predict_only and write samples to a side file:", file=sys.stderr)
            print("    lm_eval --model local-chat-completions \\", file=sys.stderr)
            print(f"        --model_args 'model={tag},base_url=...,num_concurrent=1' \\", file=sys.stderr)
            print(f"        --tasks {args.task} \\", file=sys.stderr)
            print("        --predict_only --log_samples \\", file=sys.stderr)
            print("        --use_cache <rerun-cache> \\", file=sys.stderr)
            print("        --output_path <rerun-out>/", file=sys.stderr)
            print("        # subset to unrecovered doc_ids via a task-config override", file=sys.stderr)
            print("        # or by filtering the produced samples file post-hoc.", file=sys.stderr)
            print("    # 2. Merge the rerun samples into the recovered file:", file=sys.stderr)
            print(f"    python eval/merge_samples.py --base {args.output} \\", file=sys.stderr)
            print(f"        --patch <rerun-out>/samples_{args.task}_*.jsonl \\", file=sys.stderr)
            print(f"        --doc-ids {args.unrecovered_out or '<doc_id_file>'} \\", file=sys.stderr)
            print("        --output <final>.jsonl", file=sys.stderr)
            print("    # 3. Rescore offline (no server needed):", file=sys.stderr)
            print(f"    lm_eval --tasks {args.task} \\", file=sys.stderr)
            print("        --predict_only=False --log_samples \\", file=sys.stderr)
            print("        --offline_samples <final>.jsonl \\", file=sys.stderr)
            print("        --output_path <final-out>/", file=sys.stderr)
    else:
        print("[+] Tier 2 — no remaining empties; full recovery possible")

    # Write
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            for r in samples:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[+] wrote {args.output}")
    else:
        print("[+] dry-run; no output written")

    return 0


if __name__ == "__main__":
    sys.exit(main())
