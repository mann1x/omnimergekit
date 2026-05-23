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
import re
import sys
from pathlib import Path

# Failed-round / archived result dirs left behind by retries
# (e.g. "arc_challenge_full_round1_failed_20260523", "lcb_..._round3_pyfail").
# These are NOT real benches — never surface them as table columns.
_SKIP_DIR_RE = re.compile(r"(_round\d+|_pyfail|_failed_\d{8})")


# Stable column order — matches the MicroCoder/Gemma card tables on HF.
BENCH_ORDER = [
    "humaneval_full",
    "humanevalplus_full",
    "mbpp_full",
    "lcb_medium_55",
    "lcb_medium_55_v4",
    "lcb_medium_30",
    "gsm8k_100",
    "math500_100",
    "aime_30",
    "mmlu_pro_200",
    "gpqa_diamond_full",
    "arc_challenge_full",
    "ifeval_100",
]

# Pretty-print labels for the markdown header.
BENCH_LABEL = {
    "humaneval_full": "HE",
    "humanevalplus_full": "HE+",
    "mbpp_full": "MBPP",
    "lcb_medium_55": "LCB-55",
    "lcb_medium_55_v4": "LCB-55",
    "lcb_medium_30": "LCB-30",
    "gsm8k_100": "GSM8K(100)",
    "math500_100": "MATH500(100)",
    "aime_30": "AIME(30)",
    "mmlu_pro_200": "MMLU-Pro(200)",
    "gpqa_diamond_full": "GPQA-D",
    "arc_challenge_full": "ARC-C",
    "ifeval_100": "IFEval(100)",
}


def load_summary(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def fmt_score(summary: dict) -> str:
    """Best-effort extraction of a single headline number from summary.json.

    omk_eval writes a canonical headline `score` (already the correct metric —
    flexible-extract / math_verify / pass@1,extract_chat / pass_at_1). Trust
    that first; only fall back to re-deriving from raw `results` for legacy
    summary.json files that predate the `score` field. The fallback NEVER
    prefers strict-match / exact_match,none over flexible-extract / math_verify
    — that mismatch is exactly what misreported GPQA 1.52% / math500 41% on
    2026-05-23 when the real scores were 72.73% / 94%."""
    if "_error" in summary:
        return "ERR"
    s = summary
    # Canonical headline score (current omk_eval shape).
    if isinstance(s.get("score"), (int, float)):
        v = s["score"]
        return f"{v*100:.2f}" if v <= 1.0 else f"{v:.2f}"
    # LCB (custom runner shape).
    if "pass_at_1" in s:
        v = s["pass_at_1"]
        return f"{v*100:.2f}" if v <= 1.0 else f"{v:.2f}"
    # Legacy: re-derive from raw results dict, correct-metric-first.
    res = s.get("results", {})
    PREF = ("pass@1,extract_chat", "pass@1,create_test", "exact_match,flexible-extract",
            "math_verify,none", "prompt_level_strict_acc,none", "acc_norm,none",
            "acc,none", "exact_match,strict-match", "exact_match,none")
    for pref in PREF:
        for k, v in res.items():
            if k.endswith(pref) and isinstance(v, (int, float)):
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
        # Each subdir corresponds to one template (bench). summary.json lives
        # either directly under the bench dir (flat, legacy) or one level
        # deeper under a per-served-name subdir (current omk layout):
        #   <variant>/<bench>/summary.json
        #   <variant>/<bench>/<served>/summary.json
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            if _SKIP_DIR_RE.search(sub.name):
                continue
            summ = sub / "summary.json"
            if not summ.exists():
                cand = sorted(sub.glob("*/summary.json"))
                if not cand:
                    continue
                summ = cand[-1]
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
