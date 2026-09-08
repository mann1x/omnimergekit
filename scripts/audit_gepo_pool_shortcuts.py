#!/usr/bin/env python3
"""Shortcut audit for a GEPO driver pool -- the checks build_gepo_efficiency_pool's
own audit() does NOT make.

audit() catches gold position bias, gold-is-shortest/longest, and domain skew. It
does not catch:

  1. duplicate or near-duplicate QUESTIONS (an item re-authored across blocks)
  2. a string used as the correct answer in one item and a distractor in another
  3. LEXICAL shortcuts -- a surface rule that beats chance without reading the
     question, e.g. "pick the option starting with a decisive verb" or "pick the
     one that is not a read/search/survey"

(3) is the one that bites. Measured 2026-09-08 on the 128-row driver pool BEFORE
remediation, "starts with an action verb" scored 63.4% against a 25% baseline,
because 66% of items had every distractor phrased as a read or a repeat. That is
the pool teaching option shape, not policy -- the same failure mode as
"pick the shortest", which audit() already refuses.

The fix is not to delete the correlation (acting rather than re-acquiring IS the
policy) but to add distractors that are genuine ACTIONS ON THE TARGET and still
wrong: over-action ("rewrite the module"), wrong target ("change every caller"),
or a null action dressed up ("add a catch-all handler"). After 33 such swaps the
rule fell to 44.9%.

NOTE the leak was NOT introduced by the 2026-09-08 block -- 77% of the ORIGINAL
authored items had zero action-verb distractors, against 62% of the mined ones.

Usage:  python3 -m scripts.audit_gepo_pool_shortcuts <pool.jsonl> [--max 0.60]
Exit 1 if any rule beats --max.
"""
import argparse, collections, itertools, json, re, sys

ACT = {"change","fix","run","stop","make","use","open","correct","abandon","accept","record",
       "test","time","validate","launch","edit","apply","continue","go","note","rewrite","revert",
       "roll","add","wrap","commit","rebuild","disable","replace","raise","patch","restart",
       "refactor","remove","extend","tidy","move","optimise"}
RE_ACQ = re.compile(r"^(read|re-read|search|list|survey|review|re-open|study|audit)", re.I)
RE_AGAIN = re.compile(r"\b(again|once more|a second time|re-run|re-verify|to be sure|"
                      r"to be safe|to be certain|confirm)\b", re.I)


def gold_of(r):
    return r["meta"]["choices"]["ABCD".index(r["gold"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--max", type=float, default=0.60,
                    help="fail if any surface rule scores above this (chance is 0.25)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.pool) if l.strip()]
    drv = [r for r in rows if (r["meta"].get("length_lambda") or 0) > 0]
    if not drv:
        sys.exit("no driver rows (length_lambda > 0) in this pool")
    print(f"driver rows: {len(drv)}   chance = 25.0%")
    bad = 0

    qs = [r["prompt"].split("Choices:")[0].strip() for r in drv]
    dups = [q for q, c in collections.Counter(qs).items() if c > 1]
    print(f"\nexact duplicate questions: {len(dups)}")
    bad += bool(dups)
    tk = [set(re.findall(r"[a-z]{4,}", q.lower())) for q in qs]
    near = [(i, j) for i, j in itertools.combinations(range(len(qs)), 2)
            if tk[i] and tk[j] and len(tk[i] & tk[j]) / len(tk[i] | tk[j]) > 0.70]
    print(f"near-duplicate question pairs (Jaccard > 0.70): {len(near)}")
    for i, j in near[:5]:
        print(f"   {qs[i][:72]}\n   {qs[j][:72]}\n")
    bad += bool(near)

    corr = {gold_of(r) for r in drv}
    wrong = {o for r in drv for i, o in enumerate(r["meta"]["choices"])
             if i != "ABCD".index(r["gold"])}
    clash = corr & wrong
    print(f"strings used as BOTH correct and distractor: {len(clash)}")
    bad += bool(clash)

    def score(rule):
        hit = tot = 0
        for r in drv:
            gi = "ABCD".index(r["gold"])
            picks = [i for i, o in enumerate(r["meta"]["choices"]) if rule(o)]
            if not picks:
                continue
            tot += 1
            hit += (gi in picks) / len(picks)
        return (hit / tot if tot else 0.0), tot

    print("\nsurface rules (a rule near chance means the question must be read):")
    for name, rule in [
        ("starts with an action verb", lambda o: o.split()[0].lower().strip(";,") in ACT),
        ("not a read/search/survey", lambda o: not RE_ACQ.match(o)),
        ("contains no repeat-cue word", lambda o: not RE_AGAIN.search(o)),
        ("both combined", lambda o: not RE_ACQ.match(o) and not RE_AGAIN.search(o)),
    ]:
        p, tot = score(rule)
        flag = ""
        if p > a.max:
            flag = f"  <-- FAIL (> {a.max:.0%})"
            bad += 1
        print(f"  {name:32s} {p:6.1%}  (applies to {tot}/{len(drv)}){flag}")

    z = sum(1 for r in drv
            if not any(o.split()[0].lower().strip(";,") in ACT
                       for i, o in enumerate(r["meta"]["choices"])
                       if i != "ABCD".index(r["gold"])))
    print(f"\nitems whose every distractor is a read/repeat: {z}/{len(drv)} ({100*z/len(drv):.0f}%)")
    print(f"\n{'FAIL' if bad else 'PASS'}: {bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
