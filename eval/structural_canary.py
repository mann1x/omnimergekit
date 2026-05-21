"""Structural canary — model-agnostic post-processing rules over lm-eval samples_*.jsonl.

Rules are calibrated from empirical survey of known-good and known-bad
(model × stack × bench) combinations. They catch generation/parser/scorer
breakage without needing reference scores. For score-quality regressions
(e.g. AIME 36 vs 73 with normal response structure), use the anchor-bench
layer instead.

Usage:
    structural_canary.py <samples.jsonl> [--bench-kind short_answer|thinking_reasoning]
    structural_canary.py --dir <eval_results_dir>                    # walk all samples_*.jsonl
    structural_canary.py --apply-to summary.json                     # in-place add canary block

Empirical bands (2026-05-21 calibration):
    Stack v1 (no reasoning parser, thinking=off, samples="content only"):
      healthy IFEval / ARC / HE   p10 ~120,   p50 ~840,    p99 ~14k
      healthy AIME / GPQA (think) p10 ~1400,  p50 ~21k,    p99 ~50k
      broken   v6 IFEval (parser) p10  4106,  p50  23528,  p99 87k   ← p50 ~30× normal

    Stack v2 (Fix-A in lm-eval, samples="reasoning + content"):
      Fix-A's parse_generations concatenates reasoning_content + "\n" + content
      so samples_*.jsonl resps now include the chain-of-thought even on
      short_answer benches. Thinking-on IFEval / ARC sit around p50 ~22k,
      p99 ~75k. Thresholds widened accordingly. The marker_leak_in_content
      rule (0 < channel/reasoning sentinel tokens) is what now disambiguates
      a real parser break from "model just thought a lot".
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

MARKERS = re.compile(r"<\|channel>|<channel\|>|<\|reasoning\|>|<bos>|<eos>|<\|im_start\|>(?:analysis|thinking)")

# Per-bench-kind invariants. Bench-kind is inferred from samples task name
# (override with --bench-kind) — bench-kind drives only thresholds, not logic.
BENCH_KINDS = {
    "short_answer": dict(
        # IFEval, ARC, HumanEval (chat), MBPP, GSM8K.
        # With Fix-A active the canary measures reasoning+content concatenated,
        # so thinking-on runs can hit ~22k p50 even when the parser is correct.
        # marker_leak_in_content is the rule that catches an actual parser break.
        max_p10=8000, max_p50=30000, min_p10=0,
        max_finish_length_rate=0.30,
        min_reasoning_share=0.0,
    ),
    "thinking_reasoning": dict(
        # AIME, GPQA, MATH-500, LCB-medium — thinking-on, longer content expected.
        max_p10=60000, max_p50=60000, min_p10=200,
        max_finish_length_rate=0.30,
        min_reasoning_share=0.0,
    ),
}

# Map known lm-eval task names → bench-kind
TASK_KIND = {
    "ifeval": "short_answer", "ifeval_100": "short_answer",
    "arc_challenge": "short_answer", "arc_challenge_full_chat": "short_answer", "arc_challenge_chat": "short_answer",
    "humaneval": "short_answer", "humaneval_chat": "short_answer",
    "humaneval_plus": "short_answer", "humaneval_plus_chat": "short_answer",
    "mbpp": "short_answer", "gsm8k": "short_answer",
    "aime24": "thinking_reasoning", "aime24_chat": "thinking_reasoning",
    "gpqa_diamond_cot_zeroshot": "thinking_reasoning",
    "minerva_math500": "thinking_reasoning", "math500": "thinking_reasoning",
    "lcb_medium": "thinking_reasoning",
}

@dataclass
class Rule:
    name: str
    passed: bool
    observed: object
    threshold: object
    note: str = ""

@dataclass
class CanaryReport:
    samples_file: str
    n: int
    bench_kind: str
    task: str
    rules: list = field(default_factory=list)
    passed: bool = True

    def to_dict(self):
        d = asdict(self)
        d["rules"] = [asdict(r) for r in self.rules]
        return d

def _quant(xs, q):
    if not xs:
        return 0
    if len(xs) < 10:
        return sorted(xs)[max(0, min(len(xs)-1, int(q*(len(xs)-1))))]
    return int(statistics.quantiles(xs, n=100)[max(0, int(q*100)-1)])

def _infer_task(samples_path: Path) -> str:
    m = re.search(r"samples_([a-zA-Z0-9_]+?)_\d{4}-\d{2}-\d{2}", samples_path.name)
    return m.group(1) if m else ""

def _infer_kind(task: str) -> str:
    # Strip _chat / _shadow / _full / _100 / _30 / _v4 / _reeval24k suffixes when looking up
    key = task
    for suf in ("_full", "_100", "_30", "_v4", "_reeval24k", "_55"):
        if key.endswith(suf):
            key = key[: -len(suf)]
    return TASK_KIND.get(key, TASK_KIND.get(task, "short_answer"))

def analyze(samples_path: Path, bench_kind: Optional[str] = None) -> CanaryReport:
    task = _infer_task(samples_path)
    kind = bench_kind or _infer_kind(task)
    cfg = BENCH_KINDS[kind]

    lens: list[int] = []
    empty = 0
    marker_leak = 0
    n = 0
    with open(samples_path) as f:
        for line in f:
            s = json.loads(line)
            r = s.get("resps", [[""]])[0]
            if isinstance(r, list):
                r = r[0] if r else ""
            r = r or ""
            n += 1
            lens.append(len(r))
            if not r.strip():
                empty += 1
            if MARKERS.search(r):
                marker_leak += 1

    p10 = _quant(lens, 0.10)
    p50 = int(statistics.median(lens)) if lens else 0
    p99 = _quant(lens, 0.99)
    empty_rate = empty / n if n else 0
    leak_rate = marker_leak / n if n else 0

    rules: list[Rule] = []
    rules.append(Rule("empty_content_rate", empty_rate <= 0.05, f"{empty_rate:.2%}", "<= 5%"))
    rules.append(Rule("marker_leak_in_content", leak_rate == 0,
                      f"{marker_leak}/{n} ({leak_rate:.1%})", "== 0",
                      note="catches reasoning/channel tokens leaked from parser into content"))
    rules.append(Rule("response_p10_chars",
                      cfg["min_p10"] <= p10 <= cfg["max_p10"],
                      p10, f"{cfg['min_p10']} <= p10 <= {cfg['max_p10']}",
                      note=f"healthy {kind} band"))
    rules.append(Rule("response_p50_chars",
                      p50 <= cfg["max_p50"],
                      p50, f"<= {cfg['max_p50']}",
                      note="parser dumping reasoning into content blows this up"))
    rules.append(Rule("response_p99_chars",
                      p99 <= cfg["max_p50"] * 4,
                      p99, f"<= {cfg['max_p50'] * 4}",
                      note="catches finish_reason=length cap saturation"))

    canary = CanaryReport(str(samples_path), n, kind, task, rules)
    canary.passed = all(r.passed for r in rules)
    return canary

def fmt_report(c: CanaryReport, color: bool = True) -> str:
    OK, FAIL = ("\033[32mOK\033[0m", "\033[31mFAIL\033[0m") if color and sys.stdout.isatty() else ("OK", "FAIL")
    lines = [f"== {c.samples_file}",
             f"   task={c.task} kind={c.bench_kind} n={c.n}"]
    for r in c.rules:
        tag = OK if r.passed else FAIL
        lines.append(f"   [{tag}] {r.name}: observed={r.observed}  threshold={r.threshold}"
                     + (f"  ({r.note})" if r.note else ""))
    lines.append(f"   VERDICT: {OK if c.passed else FAIL}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="samples_*.jsonl path(s)")
    ap.add_argument("--dir", help="recurse a results dir, find all samples_*.jsonl")
    ap.add_argument("--bench-kind", choices=list(BENCH_KINDS.keys()),
                    help="override bench-kind inference")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--apply-to-summary", action="store_true",
                    help="for each samples file, locate sibling summary.json and inject canary block (in-place)")
    args = ap.parse_args()

    paths: list[Path] = [Path(p) for p in args.paths]
    if args.dir:
        paths.extend(sorted(Path(args.dir).rglob("samples_*.jsonl")))
    if not paths:
        ap.error("supply samples paths or --dir")

    reports = [analyze(p, args.bench_kind) for p in paths]
    exit_code = 0 if all(r.passed for r in reports) else 2

    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        for r in reports:
            print(fmt_report(r))

    if args.apply_to_summary:
        for r in reports:
            sp = Path(r.samples_file)
            # summary.json sits at parent.parent.parent of samples_*.jsonl in our layout
            for cand in [sp.parent.parent.parent / "summary.json",
                         sp.parent / "summary.json",
                         sp.parent.parent / "summary.json"]:
                if cand.exists():
                    d = json.loads(cand.read_text())
                    d.setdefault("structural_canary", {})[r.task] = r.to_dict()
                    cand.write_text(json.dumps(d, indent=2))
                    break

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
