#!/usr/bin/env python3
"""Stage 2 of the CoderX brevity distillation (#845): turn two verified
generation files into preference pairs where the ONLY intended difference is
length.

    chosen   = a CORRECT, SHORT solution
    rejected = a CORRECT, LONG  solution to the SAME problem

Correctness is held constant on both sides by construction. That is the whole
point: a pair of (correct, incorrect) teaches correctness, and a pair of
(short, long) where the short one is wrong teaches the model to give up. Only
problems that BOTH models solved can produce a pair, which is also why the
yield is well under the pool size.

TWO PAIR SETS ARE EMITTED, NOT ONE
----------------------------------
`--out-cross` (what was asked for): chosen from the TEACHER (coder+MPE),
rejected from the STUDENT (CoderX). The teacher supplies a brevity target the
student does not reach on its own.

`--out-onpolicy` (the control, free from the same generation run): BOTH sides
drawn from the student — its shortest correct vs its longest correct sample.

The control exists because the cross set has a confound that cannot be removed
by filtering: teacher and student are different models, so their solutions
differ in algorithm, naming, and prose style, not only in length. A preference
model fitted on the cross set can improve by imitating the teacher's STYLE
while learning nothing about brevity. The on-policy set has no style axis at
all — same weights, same decoder, same distribution, only the length differs —
so it is the arm that isolates the effect. Train on the cross set if that is
the intent, but keep the control: if the cross-trained model gains and the
on-policy-trained one does not, the gain was style transfer, and style transfer
from a model with IFEval 70.0 is not something to ship blind.

WHICH REJECTED SAMPLE
---------------------
`--rejected-pick median` (default) takes the student's MEDIAN-length correct
sample. Taking the longest inflates the measured gap and trains against a tail
the student rarely produces; taking the shortest understates it. The median is
what the student actually does on a problem it can solve.

THE GAP GATE
------------
A pair whose two sides are already the same length teaches nothing and
contributes near-zero gradient; a pair where the "rejected" is SHORTER than the
"chosen" teaches verbosity outright. Both are dropped by
`--min-gap` (default 0.25 = rejected must be >=1.25x chosen). The realized dose
-- how many pairs survived and at what ratio -- is printed, because a pair count
without its length distribution is not a measurement of anything.
[[feedback_realized_dose_is_a_property_of_the_exclusion_set]]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        sys.exit(f"REFUSE: missing {p}")
    out = []
    with p.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def basis_key(r: dict) -> tuple:
    b = r.get("basis") or {}
    s = b.get("sampler") or {}
    return (b.get("sampler_name"), b.get("max_tokens"), b.get("thinking_budget"),
            s.get("temperature"), s.get("top_p"), s.get("top_k"))


def by_problem(rows: list[dict]) -> dict[str, list[dict]]:
    d = defaultdict(list)
    for r in rows:
        d[r["task_id"]].append(r)
    return d


def pick(cands: list[dict], how: str) -> dict:
    """Select one correct sample by length. cands must be non-empty."""
    s = sorted(cands, key=lambda r: r["completion_tokens"] or 0)
    if how == "shortest":
        return s[0]
    if how == "longest":
        return s[-1]
    return s[len(s) // 2]          # median


def make_pair(prob: dict, chosen: dict, rejected: dict, kind: str) -> dict:
    ct = chosen["completion_tokens"] or 0
    rt = rejected["completion_tokens"] or 0
    return {
        "task_id": prob["id"],
        "kind": kind,
        "prompt": prob["prompt"],
        "chosen": {"reasoning": chosen["reasoning"], "content": chosen["content"],
                   "tokens": ct, "model": chosen["model"],
                   "sample_idx": chosen["sample_idx"]},
        "rejected": {"reasoning": rejected["reasoning"], "content": rejected["content"],
                     "tokens": rt, "model": rejected["model"],
                     "sample_idx": rejected["sample_idx"]},
        "chosen_tokens": ct,
        "rejected_tokens": rt,
        "ratio": (rt / ct) if ct else None,
        "difficulty": (prob.get("meta") or {}).get("difficulty"),
    }


def build(pool: dict, src_chosen: dict, src_rejected: dict, *,
          kind: str, chosen_pick: str, rejected_pick: str,
          min_gap: float, same_source: bool) -> tuple[list[dict], dict]:
    pairs, census = [], defaultdict(int)
    for tid, prob in pool.items():
        # "no rows at all" is NOT the same finding as "rows exist, none passed".
        # The first means the problem was never attempted (a --limit smoke, or a
        # generation run that died partway); the second is a real capability
        # statement about the model. Collapsing them into one drop reason makes
        # an incomplete run look like a failure rate, which is exactly how the
        # 2026-08-23 4-problem smoke printed "186 never solved" for 186 problems
        # nothing had been generated for. Count them apart.
        # [[feedback_locate_the_zero_on_the_lifecycle]]
        all_c, all_r = src_chosen.get(tid, []), src_rejected.get(tid, [])
        if not all_c or not all_r:
            census["skip_not_attempted"] += 1
            continue
        cc = [r for r in all_c if r.get("passed")]
        rc = [r for r in all_r if r.get("passed")]
        if same_source:
            if len(cc) < 2:
                census["drop_need_2_correct_samples"] += 1
                continue
            ch, rj = pick(cc, "shortest"), pick(cc, "longest")
        else:
            if not cc:
                census["drop_chosen_side_never_solved"] += 1
                continue
            if not rc:
                census["drop_rejected_side_never_solved"] += 1
                continue
            ch, rj = pick(cc, chosen_pick), pick(rc, rejected_pick)
        ct, rt = ch["completion_tokens"] or 0, rj["completion_tokens"] or 0
        if ct <= 0 or rt <= 0:
            census["drop_missing_token_count"] += 1
            continue
        if rt < ct * (1.0 + min_gap):
            census["drop_gap_too_small"] += 1
            continue
        census["kept"] += 1
        pairs.append(make_pair(prob, ch, rj, kind))
    return pairs, census


def report(name: str, pairs: list[dict], census: dict, pool_n: int) -> None:
    print(f"\n=== {name} ===")
    attempted = pool_n - census.get("skip_not_attempted", 0)
    print(f"  pool problems      : {pool_n}"
          + (f"   (ATTEMPTED {attempted}; the rest have no rows on one or both "
             f"sides and are NOT a failure rate)"
             if census.get("skip_not_attempted") else ""))
    for k in sorted(census):
        if k not in ("kept", "skip_not_attempted"):
            print(f"  {k:32s}: {census[k]}")
    print(f"  {'kept':32s}: {census.get('kept', 0)} / {attempted} attempted")
    if not pairs:
        print("  NO PAIRS -- nothing to train on.")
        return
    ratios = [p["ratio"] for p in pairs if p["ratio"]]
    ct = [p["chosen_tokens"] for p in pairs]
    rt = [p["rejected_tokens"] for p in pairs]
    print(f"  chosen tokens   p50 : {st.median(ct):.0f}   "
          f"(min {min(ct)} max {max(ct)})")
    print(f"  rejected tokens p50 : {st.median(rt):.0f}   "
          f"(min {min(rt)} max {max(rt)})")
    print(f"  length ratio    p50 : {st.median(ratios):.2f}x   "
          f"(min {min(ratios):.2f} max {max(ratios):.2f})")
    diff = defaultdict(int)
    for p in pairs:
        diff[p["difficulty"] or "?"] += 1
    print(f"  by difficulty       : {dict(diff)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(REPO / "eval/lcb/lcb_rl_pool.jsonl"))
    ap.add_argument("--teacher", required=True, help="teacher generations JSONL (short)")
    ap.add_argument("--student", required=True, help="student generations JSONL (CoderX)")
    ap.add_argument("--out-cross", required=True)
    ap.add_argument("--out-onpolicy", required=True)
    ap.add_argument("--chosen-pick", default="shortest",
                    choices=["shortest", "median", "longest"])
    ap.add_argument("--rejected-pick", default="median",
                    choices=["shortest", "median", "longest"])
    ap.add_argument("--min-gap", type=float, default=0.25)
    ap.add_argument("--allow-mixed-basis", action="store_true")
    args = ap.parse_args()

    pool = {p["id"]: p for p in load_jsonl(Path(args.pool))}
    tea = load_jsonl(Path(args.teacher))
    stu = load_jsonl(Path(args.student))

    # A pair drawn from two different sampling bases is not a length contrast,
    # it is a basis contrast wearing one. Refuse by default.
    bt, bs = {basis_key(r) for r in tea}, {basis_key(r) for r in stu}
    if len(bt) != 1 or len(bs) != 1 or bt != bs:
        print(f"BASIS MISMATCH\n  teacher: {bt}\n  student: {bs}")
        if not args.allow_mixed_basis:
            print("REFUSE (pass --allow-mixed-basis to override)")
            return 3
    else:
        print(f"basis OK (both sides): {bt.pop()}")

    unver = sum(1 for r in tea + stu if r.get("passed") is None)
    if unver:
        print(f"REFUSE: {unver} rows are unverified (passed=null). "
              f"Re-run brevity_gen.py without --skip-verify first.")
        return 4

    t_by, s_by = by_problem(tea), by_problem(stu)

    cross, c_census = build(pool, t_by, s_by, kind="cross",
                            chosen_pick=args.chosen_pick,
                            rejected_pick=args.rejected_pick,
                            min_gap=args.min_gap, same_source=False)
    onpol, o_census = build(pool, s_by, s_by, kind="onpolicy",
                            chosen_pick="shortest", rejected_pick="longest",
                            min_gap=args.min_gap, same_source=True)

    report(f"CROSS  (chosen={args.chosen_pick} teacher / "
           f"rejected={args.rejected_pick} student)", cross, c_census, len(pool))
    report("ONPOLICY control (student shortest vs student longest)",
           onpol, o_census, len(pool))

    for path, rows in ((args.out_cross, cross), (args.out_onpolicy, onpol)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(rows)} pairs -> {path}")

    print("\nBREVITY_PAIRS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
