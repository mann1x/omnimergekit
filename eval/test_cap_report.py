"""Deterministic unit-smoke for cap_report(): rows of EXACT token length, all regimes.

The check must behave correctly whatever the thinking config is:
  SPLIT  reasoning in a sidecar -> the visible ceiling is max_gen - thinking_budget
  INLINE reasoning in the content -> the ceiling is the full max_gen; applying the split
         ceiling here would be a FALSE POSITIVE, which case 5 guards
  NONE   no budget at all -> only max_gen
and it must still catch a cap that no config declares (case 6), because the config has
already been wrong once (the runner overrode the template's budget for months).

Two further guards on what gets MEASURED and when a verdict may be asserted at all:
  case 7  the length must come from the raw generation, not from post-filter text that a
          task assembled (HumanEval builds a runnable program around the completion)
  case 8  the undeclared-cap branch is shape-only, so below a row floor it is reported but
          never promoted to a verdict -- equal lengths on 3 rows are a coincidence
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.environ.get("OMK_EVAL_DIR", "/shared/dev/omnimergekit/eval"))
from omk_eval import cap_report  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

TOK = AutoTokenizer.from_pretrained("unsloth/gemma-4-E2B-it")
FILLER = TOK(("The answer requires careful reasoning about the given constraints. " * 4000),
             add_special_tokens=False).input_ids


def exact(ntok: int, prefix: str = "") -> str:
    """Text that re-tokenizes to EXACTLY ntok tokens (prefix included in the count)."""
    pre = len(TOK(prefix, add_special_tokens=False).input_ids) if prefix else 0
    body = TOK.decode(FILLER[:max(ntok - pre, 1)])
    txt = prefix + body
    got = len(TOK(txt, add_special_tokens=False).input_ids)
    guard = 0
    while got != ntok and guard < 40:
        body = TOK.decode(FILLER[:max(len(TOK(body, add_special_tokens=False).input_ids)
                                      - (got - ntok), 1)])
        txt = prefix + body
        got = len(TOK(txt, add_special_tokens=False).input_ids)
        guard += 1
    return txt


def build(path: Path, spec, prefix=""):
    with open(path, "w") as fh:
        for i, (n, c) in enumerate(spec):
            fh.write(json.dumps({"doc_id": i, "filtered_resps": [exact(n, prefix)],
                                 "exact_match": c}) + "\n")


TIGHT = {"name": "t_tight", "generation": {"max_gen_toks": 16384,
                                           "thinking_token_budget": 12288}}   # allow 4096
WIDE = {"name": "t_wide", "generation": {"max_gen_toks": 65536,
                                         "thinking_token_budget": 49152}}     # allow 16384
NOBUD = {"name": "t_nobud", "generation": {"max_gen_toks": 4096,
                                           "thinking_token_budget": 0}}
HUGE = {"name": "t_huge", "generation": {"max_gen_toks": 65536,
                                         "thinking_token_budget": 0}}

SPEC = [(4093, 0.0), (4096, 0.0), (4050, 0.0), (512, 1.0), (1800, 1.0), (16384, 0.0)]
fails = []


def check(label, rep, wants):
    print(f"  {label}")
    for k, want in wants:
        got = rep.get(k)
        ok = got == want
        print(f"   {'OK  ' if ok else 'FAIL'} {k}: got={got!r} want={want!r}")
        if not ok:
            fails.append(f"{label}.{k}: {got!r} != {want!r}")


with tempfile.TemporaryDirectory() as td:
    def log(m):
        print("   " + m)

    sp = Path(td) / "s.jsonl"
    build(sp, SPEC)
    print("=" * 100)
    check("CASE 1 split 16384/12288 (allow 4096): 4093+4096 on the band, 16384 on max_gen",
          cap_report(sp, TIGHT, "unsloth/gemma-4-E2B-it", "exact_match", log),
          [("verdict", "CAPPED"), ("reasoning_regime", "split"), ("answer_allowance", 4096),
           ("ceiling_hit", "max_gen_toks,answer_allowance"), ("capped_total", 3),
           ("capped_correct", 0), ("n", 6)])

    print("=" * 100)
    check("CASE 2 wide 65536/49152 (allow 16384): SAME rows, only 16384 is on a ceiling",
          cap_report(sp, WIDE, "unsloth/gemma-4-E2B-it", "exact_match", log),
          [("verdict", "CAPPED"), ("answer_allowance", 16384),
           ("ceiling_hit", "answer_allowance"), ("capped_total", 1)])

    print("=" * 100)
    check("CASE 3 no budget, max_gen 4096: allowance is not a ceiling at all",
          cap_report(sp, NOBUD, "unsloth/gemma-4-E2B-it", "exact_match", log),
          [("answer_allowance", None), ("ceiling_hit", "max_gen_toks"),
           ("thinking_exhausted_inferred", None)])

    print("=" * 100)
    sp2 = Path(td) / "clean.jsonl"
    build(sp2, [(120, 1.0), (300, 1.0), (900, 0.0)])
    check("CASE 4 all rows short: CLEAN, no false positive",
          cap_report(sp2, TIGHT, "unsloth/gemma-4-E2B-it", "exact_match", log),
          [("verdict", "CLEAN"), ("capped_total", 0)])

    print("=" * 100)
    # Reasoning INLINE in the content, lengths well past the split allowance (4096) but all
    # different. Under the old lower-bound test every one of these was a false CAPPED.
    sp3 = Path(td) / "inline.jsonl"
    build(sp3, [(5000, 1.0), (6200, 1.0), (7100, 0.0)], prefix="<think>reasoning</think> ")
    check("CASE 5 INLINE regime past the allowance: must be CLEAN (regression guard)",
          cap_report(sp3, TIGHT, "unsloth/gemma-4-E2B-it", "exact_match", log),
          [("reasoning_regime", "inline"), ("answer_allowance", None),
           ("ceiling_hit", None), ("verdict", "CLEAN"), ("capped_total", 0)])

    print("=" * 100)
    # A cap nobody declared: nothing matches 2000, but four rows stop dead on it. The sample is
    # 12 rows so the pile-up is above the floor for asserting an undeclared cap (see CASE 8).
    sp4 = Path(td) / "undeclared.jsonl"
    build(sp4, [(2000, 0.0)] * 4 + [(640, 1.0), (720, 1.0), (300, 1.0), (1100, 0.0),
                                    (455, 1.0), (880, 1.0), (1290, 0.0), (150, 1.0)])
    check("CASE 6 undeclared cap (pile-up at 2000, no ceiling explains it): must be CAPPED",
          cap_report(sp4, HUGE, "unsloth/gemma-4-E2B-it", "exact_match", log),
          [("verdict", "CAPPED"), ("ceiling_hit", "unexplained"), ("ties_at_max", 4),
           ("capped_total", 4), ("n", 12)])

    print("=" * 100)
    # The generation cap applies to what the MODEL emitted, so the length has to be measured on
    # `resps`. On HumanEval `filtered_resps` is the ASSEMBLED PROGRAM (prompt scaffold + the
    # generation) and is longer than the generation -- measured 2026-07-28: resps 242 chars vs
    # filtered_resps 590 on one row. Here the raw rows are 200 tokens (nowhere near a ceiling)
    # while the filtered text lands exactly on the split allowance (4096); reading the filtered
    # field would manufacture a CAPPED verdict on a run that never hit anything.
    sp5 = Path(td) / "assembled.jsonl"
    with open(sp5, "w") as fh:
        for i, n in enumerate((200, 240, 310)):
            fh.write(json.dumps({"doc_id": i, "resps": [exact(n)],
                                 "filtered_resps": [exact(4096)], "exact_match": 1.0}) + "\n")
    check("CASE 7 filtered_resps inflated by scaffold: must measure resps -> CLEAN",
          cap_report(sp5, TIGHT, "unsloth/gemma-4-E2B-it", "exact_match", log),
          [("verdict", "CLEAN"), ("capped_total", 0), ("max_completion_tokens", 310)])

    print("=" * 100)
    # The undeclared-cap branch is pure shape, with no arithmetic behind it, so on a 3-row smoke
    # a coincidence of equal lengths must NOT be promoted to a verdict. It is still surfaced in
    # ties_at_max and in a [cap-check] log line -- suppressed, not hidden.
    sp6 = Path(td) / "smallpileup.jsonl"
    build(sp6, [(2000, 1.0), (2000, 1.0), (2000, 1.0)])
    check("CASE 8 pile-up below the row floor: reported but NOT asserted",
          cap_report(sp6, HUGE, "unsloth/gemma-4-E2B-it", "exact_match", log),
          [("verdict", "CLEAN"), ("ceiling_hit", None), ("ties_at_max", 3),
           ("capped_total", 0)])

print("=" * 100)
print("UNIT-SMOKE: " + ("PASS — all 8 cases" if not fails else "FAIL\n  " + "\n  ".join(fails)))
sys.exit(1 if fails else 0)
