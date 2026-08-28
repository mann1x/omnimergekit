#!/usr/bin/env python3
"""Build the GEPO capability-replay pool -- the lambda=0 counterweight to the LCB pool.

WHY THIS EXISTS
---------------
GEPO's brevity reward has support ONLY on LCB coding problems:

    reward = (1 - lambda * min(ntok / budget, 1))  if tests pass  else 0

Every other capability is outside the objective, so nothing in the training signal
defends it. The 2026-08-26 paired degradation census on gepo2-vs-armJ (qwen_suite,
one basis, McNemar exact on the paired pass/fail) measured what that costs:

    GPQA        -7.58pp   25/10 discordant   p=0.0167   monotone gepo1->gepo2
    HumanEval   -3.66pp    6/0  discordant   p=0.0312   monotone gepo1->gepo2

Those two are the only benches that survive significance AND are monotone with dose.
LCB (-7.79pp, p=0.2101) and IFEval (-3.00pp, p=0.5078) are neither -- and the
same-model repeat draw (2026-08-26) put the LCB n=77 sampled draw-to-draw band at
13 flips / ~8pp, which is larger than the gepo2-vs-armJ gap it was meant to explain.
So this pool defends the two MEASURED losses and deliberately does not chase LCB
or IFEval numbers that have no floor under them.

LAMBDA = 0, AND THAT IS THE WHOLE POINT
---------------------------------------
Replay rows carry `meta.length_lambda = 0.0`: pure correctness gate, no length term.
Replay exists to hold capability, not to shorten. Putting the brevity term on these
rows would apply length pressure to exactly the reasoning the LCB arm was already
eroding. The LCB rows keep their own lambda (0.7 as run1/run2/run3 shipped it), so
the mixed pool trains "be short on code" and "stay correct on science/functions"
as two separate contracts rather than one averaged one.

CONTAMINATION -- BOTH TIERS ARE EVAL SETS IN THIS REPO
------------------------------------------------------
This is not the usual "pick a training set" problem. Both source datasets ARE
benches we score:

  * GPQA: the frozen eval is Diamond (198). Verified here: Diamond is a strict
    subset of Main (198/198 by Record ID AND 198/198 by normalised question hash,
    and the two exclusions agree on all 448 rows). Main-minus-Diamond = 250.
  * MBPP: the frozen eval is `mbpp_full` = the `full` config's **test** split (500),
    per eval/templates/mbpp_full.yaml. It is scored 3-shot, and the exemplars are
    NOT drawn at random -- lm-eval's `list_fewshot_samples` pins MBPP task_ids
    2/3/4. Training on those would put the eval's own prompt prefix into the
    weights, so they are held out too even though they are not test rows.

HumanEval itself contributes NO training rows: all 164 are the eval. MBPP is the
donor for that capability because it is the same skill (write a Python function to
a spec, graded by unit tests) on disjoint problems. The HE overlap check below is a
measurement, not an assumption -- it refuses on a nonzero intersection.

The gate is modelled on build_lcb_rl_pool.py: a single surviving holdout id makes
every downstream cell of that bench a lie, so it exits 3 rather than warning.

OUTPUT SCHEMA
-------------
A superset of the LCB pool's {id, source, prompt, gold, meta}, so the existing
launcher reads it unchanged, plus two fields that make each row declare its own
contract instead of inheriting a global one:

    meta.reward_kind    "mc_letter" | "mbpp_exec"   -- which verifier scores it
    meta.length_lambda  0.0                          -- no brevity term on replay

SCOPE: this script builds the dataset. Wiring the two new verifiers into the
trainer is a separate change -- gepo_brevity.py today knows only the LCB executor.
A pool row whose reward_kind has no registered verifier must make the trainer
REFUSE, not silently score 0.0, or the replay arm becomes an all-zero-reward tier
that quietly drags the policy instead of defending it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import random
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

HUB = "/root/.cache/huggingface/hub"


def resolve_snapshot(repo_dir: str, must_contain: list[str]) -> str:
    """Find the snapshot that actually HAS the files, rather than hardcoding a hash.

    The hashes were hardcoded here originally, which is wrong twice over: they differ
    between hosts, and -- measured on bs2 2026-08-28 -- a host can carry the SAME hash
    while holding only a SUBSET of the files. bs2 had gpqa_diamond.csv but not
    gpqa_main.csv, because the eval only ever needed Diamond. A hash equality check
    would have passed there and then failed deep inside the build, or silently produced
    a different pool. So select on the presence of the files this build actually reads,
    and name what is missing if nothing qualifies."""
    root = pathlib.Path(HUB) / repo_dir / "snapshots"
    if not root.is_dir():
        sys.exit(f"REFUSE: no HF cache at {root}. Download the dataset first.")
    cands = sorted(x for x in root.iterdir() if x.is_dir())
    for snap in cands:
        if all((snap / f).is_file() for f in must_contain):
            return str(snap)
    have = {x.name: sorted(f.name for f in x.rglob("*") if f.is_file())[:8]
            for x in cands}
    sys.exit(f"REFUSE: no snapshot of {repo_dir} has all of {must_contain}.\n"
             f"  snapshots present: {have}\n"
             "  This host cannot reproduce the pool. Build it where the files exist and "
             "copy the JSONL, or download the missing file.")


GPQA_SNAP = resolve_snapshot("datasets--Idavidrein--gpqa",
                             ["gpqa_main.csv", "gpqa_diamond.csv"])
MBPP_SNAP = resolve_snapshot("datasets--google-research-datasets--mbpp",
                             ["full/test-00000-of-00001.parquet"])
HE_SNAP = str(pathlib.Path(resolve_snapshot(
    "datasets--openai--openai_humaneval",
    ["openai_humaneval/test-00000-of-00001.parquet"])) / "openai_humaneval")

# lm-eval pins these as the mbpp 3-shot exemplars (eval/tasks/_mbpp_utils.py ->
# upstream list_fewshot_samples). Resolved by running it, not by reading the docs.
MBPP_FEWSHOT_IDS = {2, 3, 4}

# Matches eval/tasks/mbpp_chat.yaml doc_to_text so replay practises the capability
# in the shape it is measured in. Legitimate precisely because the problems are
# disjoint from the scored split.
MBPP_TEMPLATE = ("You are an expert Python programmer, and here is your task:\n"
                 "{text}\nYour code should pass these tests:\n{tests}")

GPQA_TEMPLATE = ("What is the correct answer to this question: {question}\n\n"
                 "Choices:\n(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n\n"
                 "Give a step-by-step reasoned answer, then finish with "
                 '"The correct answer is (X)" where X is A, B, C or D.')


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def qhash(s: str) -> str:
    return hashlib.sha256(norm(s).encode()).hexdigest()


def code_hash(s: str) -> str:
    """Whitespace- and comment-insensitive hash, for the HumanEval overlap check."""
    body = re.sub(r"#.*", "", s or "")
    return hashlib.sha256(re.sub(r"\s+", "", body).encode()).hexdigest()


# --------------------------------------------------------------------------- GPQA
def build_gpqa(seed: int) -> tuple[list[dict], dict]:
    """Main MINUS Diamond. Diamond is the frozen eval; Main is its superset."""
    def load(name: str) -> list[dict]:
        p = pathlib.Path(GPQA_SNAP) / name
        if not p.is_file():
            sys.exit(f"REFUSE: missing {p} -- cannot prove the GPQA holdout")
        return list(csv.DictReader(p.open(encoding="utf-8")))

    main, dia = load("gpqa_main.csv"), load("gpqa_diamond.csv")
    d_ids = {r["Record ID"] for r in dia}
    d_hsh = {qhash(r["Question"]) for r in dia}

    # Two independent exclusions. If they disagree, the id column and the question
    # text disagree about what Diamond IS, and neither can be trusted as a holdout.
    by_id = {r["Record ID"] for r in main if r["Record ID"] not in d_ids}
    by_h = {r["Record ID"] for r in main if qhash(r["Question"]) not in d_hsh}
    if by_id != by_h:
        sys.exit(f"REFUSE: GPQA id-exclusion and hash-exclusion disagree on "
                 f"{len(by_id ^ by_h)} rows -- holdout is not well defined")

    rows = []
    for r in main:
        if r["Record ID"] in d_ids or qhash(r["Question"]) in d_hsh:
            continue
        opts = [r["Correct Answer"], r["Incorrect Answer 1"],
                r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        if any(not (o or "").strip() for o in opts):
            continue
        # Seed per-question from the Record ID: the shuffle is then stable under
        # row reordering and independent of how many rows survived the filter.
        order = [0, 1, 2, 3]
        random.Random(f"{seed}:{r['Record ID']}").shuffle(order)
        shuffled = [opts[i] for i in order]
        gold = "ABCD"[order.index(0)]
        rows.append({
            "id": f"gpqa_main/{r['Record ID']}",
            "source": "gpqa_main_minus_diamond",
            "prompt": GPQA_TEMPLATE.format(question=r["Question"].strip(),
                                           a=shuffled[0].strip(), b=shuffled[1].strip(),
                                           c=shuffled[2].strip(), d=shuffled[3].strip()),
            "gold": gold,
            "meta": {
                "reward_kind": "mc_letter",
                "length_lambda": 0.0,
                "choices": shuffled,
                "domain": r.get("High-level domain", ""),
                "subdomain": r.get("Subdomain", ""),
                "question_sha256": qhash(r["Question"]),
            },
        })
    return rows, {"main": len(main), "diamond": len(dia), "held_out": len(d_ids)}


# --------------------------------------------------------------------------- MBPP
def build_mbpp(min_tests: int) -> tuple[list[dict], dict]:
    """MBPP full MINUS the scored test split MINUS the pinned 3-shot exemplars."""
    import pandas as pd

    def load(split: str):
        p = pathlib.Path(MBPP_SNAP) / "full" / f"{split}-00000-of-00001.parquet"
        if not p.is_file():
            sys.exit(f"REFUSE: missing {p} -- cannot prove the MBPP holdout")
        return pd.read_parquet(p)

    test = load("test")
    holdout = {int(t) for t in test["task_id"]} | MBPP_FEWSHOT_IDS

    rows = []
    for split in ("train", "validation", "prompt"):
        df = load(split)
        for _, r in df.iterrows():
            tid = int(r["task_id"])
            if tid in holdout:
                continue
            tests = list(r["test_list"])
            if len(tests) < min_tests:
                continue
            rows.append({
                "id": f"mbpp/{tid}",
                "source": f"mbpp_full_{split}",
                "prompt": MBPP_TEMPLATE.format(text=str(r["text"]).strip(),
                                               tests="\n".join(tests)),
                "gold": "",              # unused: the verifier executes the tests
                "meta": {
                    "reward_kind": "mbpp_exec",
                    "length_lambda": 0.0,
                    "tests": tests,
                    "n_tests": len(tests),
                    "test_setup_code": str(r.get("test_setup_code") or ""),
                    "reference_code": str(r["code"]),
                    "code_sha256": code_hash(str(r["code"])),
                },
            })
    return rows, {"scored_test_split": len(test),
                  "fewshot_exemplars": sorted(MBPP_FEWSHOT_IDS),
                  "held_out": len(holdout)}


def humaneval_overlap(rows: list[dict]) -> list[str]:
    """Measured, not assumed: does any MBPP replay row restate a HumanEval problem?"""
    import pandas as pd
    p = pathlib.Path(HE_SNAP) / "test-00000-of-00001.parquet"
    if not p.is_file():
        sys.exit(f"REFUSE: missing {p} -- cannot prove the HumanEval exclusion")
    he = pd.read_parquet(p)
    he_codes = {code_hash(str(a) + str(b))
                for a, b in zip(he["prompt"], he["canonical_solution"])}
    he_codes |= {code_hash(str(b)) for b in he["canonical_solution"]}
    return [r["id"] for r in rows
            if r["meta"].get("code_sha256") in he_codes]


def check_holdout(gpqa: list[dict], mbpp: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """Re-derive every holdout set FROM SOURCE and intersect it with the built pool.

    Deliberately does not reuse the exclusion sets computed during the build: a gate
    that trusts the builder's own bookkeeping can only confirm the builder agrees
    with itself. Re-reading Diamond / the MBPP test split / HumanEval means a bug in
    the exclusion is caught by the gate rather than laundered through it.

    Split out of main() so it can be run against a POISONED pool -- see
    scripts/test_gepo_replay_gate.py. An unfired gate is not a verified gate.
    """
    import pandas as pd
    dia_ids = {f"gpqa_main/{r['Record ID']}" for r in
               csv.DictReader((pathlib.Path(GPQA_SNAP) / "gpqa_diamond.csv")
                              .open(encoding="utf-8"))}
    leaked_g = sorted({r["id"] for r in gpqa} & dia_ids)

    test_ids = {f"mbpp/{int(t)}" for t in
                pd.read_parquet(pathlib.Path(MBPP_SNAP) / "full" /
                                "test-00000-of-00001.parquet")["task_id"]}
    test_ids |= {f"mbpp/{t}" for t in MBPP_FEWSHOT_IDS}
    leaked_m = sorted({r["id"] for r in mbpp} & test_ids)
    leaked_he = humaneval_overlap(mbpp)
    return leaked_g, leaked_m, leaked_he


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "eval" / "replay" / "gepo_replay_pool.jsonl"))
    ap.add_argument("--min-tests", type=int, default=1,
                    help="drop MBPP rows with fewer tests -- the correctness gate is "
                         "only as trustworthy as the tests behind it")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gpqa, g_prov = build_gpqa(args.seed)
    print("=== Tier S: GPQA (defends the -7.58pp GPQA loss) ===")
    print(f"  main {g_prov['main']}  diamond(HELD OUT) {g_prov['diamond']}  "
          f"-> {len(gpqa)} replay rows")

    mbpp, m_prov = build_mbpp(args.min_tests)
    print("\n=== Tier P: MBPP (defends the -3.66pp HumanEval loss) ===")
    print(f"  scored test split(HELD OUT) {m_prov['scored_test_split']}  "
          f"3-shot exemplars(HELD OUT) {m_prov['fewshot_exemplars']}  "
          f"-> {len(mbpp)} replay rows")

    leaked_g, leaked_m, leaked_he = check_holdout(gpqa, mbpp)

    print("\n=== contamination gates ===")
    print(f"  GPQA Diamond ids in pool     : {len(leaked_g)}")
    print(f"  MBPP scored/fewshot in pool  : {len(leaked_m)}")
    print(f"  HumanEval solutions in pool  : {len(leaked_he)}")
    if leaked_g or leaked_m or leaked_he:
        for t in (leaked_g + leaked_m + leaked_he)[:20]:
            print(f"    LEAK {t}")
        print("\nREFUSE: holdout leaked into the replay pool -- every downstream "
              "cell of that bench would become train-on-test.")
        return 3

    rows = gpqa + mbpp
    if not rows:
        print("\nREFUSE: empty pool")
        return 4
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        print(f"\nREFUSE: {len(ids) - len(set(ids))} duplicate ids in the pool")
        return 5
    bad = [r["id"] for r in rows if r["meta"]["length_lambda"] != 0.0]
    if bad:
        print(f"\nREFUSE: {len(bad)} replay rows carry a nonzero length_lambda -- "
              "replay must be a pure correctness gate")
        return 6

    random.Random(args.seed).shuffle(rows)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    by_kind: dict[str, int] = {}
    for r in rows:
        k = r["meta"]["reward_kind"]
        by_kind[k] = by_kind.get(k, 0) + 1
    print(f"\nHOLDOUT_OK: 0 leaked ids in a {len(rows)}-prompt replay pool")
    print(f"composition by reward_kind: {by_kind}")
    print(f"lambda: 0.0 on all {len(rows)} rows (pure correctness gate)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
