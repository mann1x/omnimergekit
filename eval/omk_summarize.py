#!/usr/bin/env python3
"""Print final-table-ready summaries from one or more omk_eval results dirs.

Reads `summary.json` files written by `omk_eval.py` under
  <results-dir>/<template-name>/summary.json
and emits a single markdown table covering all (model, bench) pairs.

Used to roll up the v3/v4/128e LCB-55 comparison, and (more generally) any
multi-bench MicroCoder/Gemma run into a copy-pasteable card table.

Usage:
    omk_summarize.py <results-dir> [results-dir ...]
        Each <results-dir> typically corresponds to one model variant.

Output: markdown to stdout. Pipe to a file when used in scripts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Stable column order — matches the MicroCoder card table on HF.
BENCH_ORDER = [
    "humaneval_full",
    "humanevalplus_full",
    "mbpp_full",
    "gsm8k_100",
    "mmlu_pro_200",
    "gpqa_diamond_full",
    "aime_30",
    "lcb_medium_55",
    "lcb_medium_30",
]

# Pretty-print labels for the markdown header.
BENCH_LABEL = {
    "humaneval_full": "HE",
    "humanevalplus_full": "HE+",
    "mbpp_full": "MBPP",
    "gsm8k_100": "GSM8K(100)",
    "mmlu_pro_200": "MMLU-Pro(200)",
    "gpqa_diamond_full": "GPQA-D",
    "aime_30": "AIME(30)",
    "lcb_medium_55": "LCB-55",
    "lcb_medium_30": "LCB-30",
}


def load_summary(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def fmt_score(summary: dict) -> str:
    """Best-effort extraction of a single headline number from summary.json.

    omk_eval writes different shapes for lm-eval vs lcb runs. Order matches
    decreasing trust: explicit pass@1 → exact_match → flexible-extract."""
    if "_error" in summary:
        return "ERR"
    s = summary
    # LCB
    if "pass_at_1" in s:
        v = s["pass_at_1"]
        return f"{v*100:.2f}" if v <= 1.0 else f"{v:.2f}"
    # lm-eval HE/MBPP
    for key in ("pass@1,create_test", "pass_at_1,create_test"):
        if key in s.get("results", {}):
            return f"{s['results'][key]*100:.2f}"
    res = s.get("results", {})
    # lm-eval flat results dict
    for k, v in res.items():
        if k.endswith("exact_match,flexible-extract") or k.endswith("exact_match,strict-match"):
            return f"{v*100:.2f}"
        if k.endswith("pass@1,create_test"):
            return f"{v*100:.2f}"
    return "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--name-from", default="dir",
                    choices=("dir", "summary"),
                    help="model name source: dir basename or summary.json model field")
    args = ap.parse_args()

    # rows[model] = {bench: score_str}
    rows: dict[str, dict[str, str]] = {}
    benches_seen: set[str] = set()

    for d in args.dirs:
        if not d.exists():
            print(f"# WARN missing {d}", file=sys.stderr)
            continue
        model = d.name
        rows.setdefault(model, {})
        # Each subdir corresponds to one template.
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            summ = sub / "summary.json"
            if not summ.exists():
                continue
            data = load_summary(summ)
            if args.name_from == "summary":
                model = data.get("model", model)
                rows.setdefault(model, {})
            bench = sub.name
            benches_seen.add(bench)
            rows[model][bench] = fmt_score(data)

    if not rows:
        print("# no summaries found", file=sys.stderr)
        return 2

    # Order columns: canonical BENCH_ORDER first, then anything else seen.
    cols = [b for b in BENCH_ORDER if b in benches_seen] + \
           sorted(b for b in benches_seen if b not in BENCH_ORDER)
    headers = ["model"] + [BENCH_LABEL.get(b, b) for b in cols]

    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for model in sorted(rows):
        cells = [rows[model].get(b, "—") for b in cols]
        print(f"| {model} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
