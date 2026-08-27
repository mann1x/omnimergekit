#!/usr/bin/env python3
"""Conciseness benchmark: token cost CONDITIONED ON being right.

WHY A NEW REPORT AND NOT JUST token_stats
-----------------------------------------
summary.json already carries completion_tokens p10/p50/p90 -- but over ALL problems,
correct and incorrect together. That marginal is not a conciseness measure and can
inverte the ranking outright: a model that gives up early on the problems it cannot
solve buys a low p50 with failures, and a model that thinks hard only on the ones it
actually solves is punished for it. Conciseness is only meaningful among completions
that EARNED their tokens.

So every column here is conditioned:

  pass          n_pass / n                          the correctness anchor -- read FIRST
  tok_p50|ok    median completion tokens on PASSES  the conciseness number
  tok_p90|ok    tail on passes                      catches "usually terse, sometimes runaway"
  TCS           sum(all tokens) / n_pass            token cost per SOLVED problem
  cap%          finish_reason == length             the runaway rate

TCS is the headline. It is the only one that cannot be gamed in either direction: being
verbose raises the numerator, being wrong lowers the denominator. It answers the question
actually being asked -- "what does this model cost me per problem it actually solves".

BASIS DISCIPLINE
----------------
A conciseness number is a property of (weights x quant x sampler x caps x scorer). The
two LCB-77q cohorts on this host are NOT comparable:
    qwen_suite/lcb_v6_77q      sampler=recommended (t0.6/p0.95/k20), max_gen 32768
    ream_arms/lcb_v6_77q_48k   greedy,                               max_gen 49152
The caps alone move the cap-hit rate, and the cap-hit rate moves the token medians. This
script REFUSES to put rows from different (sampler, max_gen_toks) bases in one table
unless --allow-mixed-basis is passed, and always prints the basis it grouped on.
[[feedback_batch_composition_is_an_eval_basis]] [[feedback_sampler_is_a_cohort_fact_read_it]]

SCORER VERSION. LCB samples may carry both `passed` and `passed_b606` (the bug-604/606
re-score). When the b606 field exists it WINS, and mixing the two across arms is refused:
a scorer fix voids every earlier cell of that bench.
[[feedback_a_scorer_fix_voids_every_earlier_cell_of_that_bench]]

BUT: field presence is NOT scorer identity. A run made AFTER the fix writes a plain
`passed` produced by the FIXED scorer, and naive inference labels it pre-fix -- the exact
inversion of the truth. That is what split the 2026-08-22 OmniMerge-v4 row off from the
cohort it belonged to. Cells carrying only `passed` are therefore labelled `plain`, not
`orig`, and merging them with `b606` requires --assert-scorer-equiv, which you may pass
ONLY after running the control:

    python scripts/lcb_scorer_equiv.py <arm-with-both-columns>

and seeing SCORER_IS_B606. Pick an arm where the two banked columns actually DIFFER --
an arm where b606 changed nothing agrees with both and discriminates nothing. On the
qwen_suite cohort only qwencodermpe_q6k qualifies (delta 1/77); it returned
`current==passed_b606 77/77, current==passed 76/77` on 2026-08-22.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys


def _pct(x, n):
    return 100.0 * x / n if n else float("nan")


def load_cell(cell_dir: str) -> dict | None:
    """Read one <bench>/<served>/ cell into a row, or None if it has no usable samples."""
    summ_p = os.path.join(cell_dir, "summary.json")
    if not os.path.exists(summ_p):
        return None
    summ = json.load(open(summ_p))

    # Prefer the b606-rescored samples file when present -- it supersedes the original.
    cands = sorted(glob.glob(os.path.join(cell_dir, "*.samples.jsonl")))
    b606 = [c for c in cands if ".b606." in c]
    samples_p = (b606 or cands or [None])[0]
    if not samples_p:
        return None
    rows = []
    with open(samples_p) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    if not rows:
        return None

    # REFUSE rather than report zeros. MultiPL-E samples carry only
    # {completion, doc_id, filtered_resps, resps, task_id} -- no per-problem verdict and
    # no per-problem token count. Reading them with .get(...) or 0 yields a table of
    # `pass 0.000 / TCS inf` for EVERY arm, which looks like a finding and is noise.
    # A metric that cannot be conditioned must say so, not print a number.
    # [[feedback_a_check_gold_fails_is_a_broken_check]]
    missing = [k for k in ("completion_tokens",) if k not in rows[0]]
    if "passed" not in rows[0] and "passed_b606" not in rows[0]:
        missing.append("passed/passed_b606")
    if missing:
        return {"unusable": os.path.basename(cell_dir),
                "bench": os.path.basename(os.path.dirname(cell_dir)),
                "why": "samples lack " + ", ".join(missing)}

    # scorer field: b606 wins where it exists. `plain` != `orig` -- see the module
    # docstring; a fresh post-fix run also writes plain `passed`.
    pass_key = "passed_b606" if "passed_b606" in rows[0] else "passed"
    scorer = "b606" if pass_key == "passed_b606" else "plain"

    tok = [r.get("completion_tokens") or 0 for r in rows]
    ok = [bool(r.get(pass_key)) for r in rows]
    caps = sum(1 for r in rows if r.get("finish_reason") == "length")
    ok_tok = [t for t, p in zip(tok, ok) if p]
    n, n_pass = len(rows), sum(ok)

    sam = summ.get("sampler") or {}
    gen = summ.get("generation") or {}
    # max_gen_toks is not always echoed into summary.json (task #840); fall back to the
    # observed ceiling, which is what actually bound the run.
    max_gen = gen.get("max_gen_toks") or (max(tok) if tok else None)

    return {
        "bench": os.path.basename(os.path.dirname(cell_dir)),
        "served": os.path.basename(cell_dir),
        "root": os.path.basename(os.path.dirname(os.path.dirname(cell_dir))),
        "n": n, "n_pass": n_pass,
        "pass": n_pass / n if n else float("nan"),
        "tok_p50_ok": st.median(ok_tok) if ok_tok else float("nan"),
        "tok_p90_ok": (sorted(ok_tok)[int(0.9 * (len(ok_tok) - 1))]
                       if ok_tok else float("nan")),
        "tok_p50_all": st.median(tok) if tok else float("nan"),
        "tcs": (sum(tok) / n_pass) if n_pass else float("inf"),
        "cap_pct": _pct(caps, n),
        "sampler": sam.get("name", "?"),
        "max_gen": max_gen,
        "scorer": scorer,
        "score": summ.get("score"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+",
                    help="cell dirs or globs, e.g. /srv/ml/eval_results/*/lcb_v6_77q*/*")
    ap.add_argument("--allow-mixed-basis", action="store_true")
    ap.add_argument("--assert-scorer-equiv", action="store_true",
                    help="collapse 'plain' and 'b606' into one scorer axis. Pass ONLY "
                         "after scripts/lcb_scorer_equiv.py printed SCORER_IS_B606 on an "
                         "arm where the two banked columns DIFFER.")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    paths: list[str] = []
    for c in args.cells:
        paths.extend(sorted(glob.glob(c)) if any(ch in c for ch in "*?[") else [c])
    allrows = [r for r in (load_cell(p) for p in paths if os.path.isdir(p)) if r]
    unusable = [r for r in allrows if "unusable" in r]
    rows = [r for r in allrows if "unusable" not in r]
    if unusable:
        benches = sorted({r["bench"] for r in unusable})
        print(f"SKIPPED {len(unusable)} cell(s) on {', '.join(benches)} -- "
              f"{unusable[0]['why']}.")
        print("  This bench cannot be scored for conciseness: without a per-problem "
              "verdict and token count there is no way to condition on correctness.\n")
    if not rows:
        print("REFUSE: no cells with usable samples found")
        return 2

    # group by basis; a basis is (sampler, max_gen ceiling, scorer version)
    def basis(r):
        sc = "b606==plain(asserted)" if args.assert_scorer_equiv else r["scorer"]
        return (r["sampler"], r["max_gen"], sc)

    if args.assert_scorer_equiv:
        print("SCORER EQUIVALENCE ASSERTED by the operator: 'plain' cells are being "
              "treated as b606-scored. This is only valid if lcb_scorer_equiv.py "
              "printed SCORER_IS_B606.\n")

    groups: dict = {}
    for r in rows:
        groups.setdefault(basis(r), []).append(r)

    if len(groups) > 1 and not args.allow_mixed_basis:
        print(f"NOTE: {len(groups)} distinct bases found -- reporting them SEPARATELY.")
        print("      (pass --allow-mixed-basis to force one table; the numbers would "
              "not be comparable)\n")

    for (sampler, max_gen, scorer), grp in sorted(
            groups.items(), key=lambda kv: -len(kv[1])):
        print(f"=== basis: sampler={sampler}  max_gen={max_gen}  scorer={scorer}  "
              f"({len(grp)} cells) ===")
        print(f"{'served':30s} {'bench':18s} {'pass':>7s} {'tok_p50|ok':>11s} "
              f"{'tok_p90|ok':>11s} {'TCS':>9s} {'cap%':>6s} {'tok_p50_all':>12s}")
        for r in sorted(grp, key=lambda r: r["tcs"]):
            print(f"{r['served'][:30]:30s} {r['bench'][:18]:18s} "
                  f"{r['pass']:7.3f} {r['tok_p50_ok']:11.0f} {r['tok_p90_ok']:11.0f} "
                  f"{r['tcs']:9.0f} {r['cap_pct']:6.1f} {r['tok_p50_all']:12.0f}")
        print()

    print("TCS = sum(all completion tokens) / n_pass -- token cost per SOLVED problem. "
          "Lower is better.")
    print("Read `pass` first: a low TCS on a low pass rate is a model that is cheap "
          "because it is wrong.")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
