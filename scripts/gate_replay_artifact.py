#!/usr/bin/env python3
"""Contamination gate on the SHIPPED replay pool -- runnable on the RUN host.

WHY THIS EXISTS SEPARATELY FROM test_gepo_replay_gate.py
--------------------------------------------------------
That test is a BUILD-time gate: it re-runs build_gpqa/build_mbpp from source and
poisons the in-memory result. It therefore needs `gpqa_main.csv`, which is a build
input, and it can only run where the pool is built.

Measured on bs2 2026-08-28: the run host has `gpqa_diamond.csv` but NOT
`gpqa_main.csv`, because the eval only ever needed Diamond. So the build gate is
structurally unrunnable there -- and bs2 is precisely the host that will spend 25-30
GPU-hours training on the file. What was verified on solidpc was the builder's
output in memory; what bs2 trains on is a JSONL that arrived over scp. A sha256
match between them is a match against MY NOTE of what was audited, not a
measurement against the eval sets. [[feedback_match_the_shipped_artifact_not_your_note]]

This gate closes that: it reads the artifact as the trainer will read it, and checks
it against the three holdout sets THIS host actually has. It needs only eval-side
files (Diamond / MBPP test / HumanEval), all of which are present wherever the
benches are scored.

REFUSES rather than skips when a holdout set is absent. "I could not check" and
"I checked and it was clean" must never produce the same exit code -- a gate that
silently degrades to a no-op on the host that matters is worse than no gate, because
it is quoted as evidence. [[feedback_never_skip_silently]]

SELF-CONTROL: three zeros prove nothing on their own. Each gate is fired against a
real holdout row before the clean pool is judged, so the zeros are readable on THIS
host, not inherited from a passing run somewhere else.
[[feedback_a_zero_needs_a_nonzero_floor_control]]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
import sys

HUB = "/root/.cache/huggingface/hub"
# lcb_exec is legitimate here: the rebalanced run4 pool is ONE file carrying the
# LCB tier alongside the replay tiers, so a gate that rejected lcb_exec would
# reject the artifact it exists to certify. The invariant that actually matters is
# not "no LCB rows" but "length pressure on LCB and ONLY on LCB", checked below.
KNOWN_KINDS = {"mc_letter", "mbpp_exec", "lcb_exec"}
MBPP_FEWSHOT_IDS = {2, 3, 4}          # lm-eval's pinned 3-shot exemplars
EXPECT = {"diamond": 198, "mbpp_test": 500, "humaneval": 164}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def qhash(s: str) -> str:
    return hashlib.sha256(norm(s).encode()).hexdigest()


def code_hash(s: str) -> str:
    body = re.sub(r"#.*", "", s or "")
    return hashlib.sha256(re.sub(r"\s+", "", body).encode()).hexdigest()


def resolve(repo_dir: str, must_contain: list[str]) -> pathlib.Path:
    root = pathlib.Path(HUB) / repo_dir / "snapshots"
    if root.is_dir():
        for snap in sorted(x for x in root.iterdir() if x.is_dir()):
            if all((snap / f).is_file() for f in must_contain):
                return snap
    sys.exit(f"REFUSE: cannot locate {must_contain} under {root}. This host cannot "
             f"PROVE the pool is uncontaminated, and an unprovable pool must not be "
             f"trained on. Download the eval set, or run this gate where it exists.")


# --------------------------------------------------------------- holdout sets
def load_holdouts() -> dict:
    import pandas as pd

    dia_p = resolve("datasets--Idavidrein--gpqa", ["gpqa_diamond.csv"])
    mb_p = resolve("datasets--google-research-datasets--mbpp",
                   ["full/test-00000-of-00001.parquet"])
    he_p = resolve("datasets--openai--openai_humaneval",
                   ["openai_humaneval/test-00000-of-00001.parquet"])

    dia = list(csv.DictReader((dia_p / "gpqa_diamond.csv").open(encoding="utf-8")))
    mb = pd.read_parquet(mb_p / "full" / "test-00000-of-00001.parquet")
    he = pd.read_parquet(he_p / "openai_humaneval" / "test-00000-of-00001.parquet")

    h = {
        "gpqa_ids": {r["Record ID"] for r in dia},
        "gpqa_hashes": {qhash(r["Question"]) for r in dia},
        "mbpp_ids": {int(x) for x in mb["task_id"]} | MBPP_FEWSHOT_IDS,
        "he_hashes": {code_hash(str(x)) for x in he["canonical_solution"]},
        "n": {"diamond": len(dia), "mbpp_test": len(mb), "humaneval": len(he)},
    }
    # An empty or truncated holdout set passes every clean pool. Pin the sizes.
    for k, want in EXPECT.items():
        if h["n"][k] != want:
            sys.exit(f"REFUSE: holdout {k} has {h['n'][k]} rows, expected {want}. "
                     f"A wrong-sized holdout set cannot certify anything.")
    return h


# ------------------------------------------------------------------- the gates
def leaks(rows: list[dict], h: dict) -> tuple[list, list, list]:
    """Returns (gpqa_leaks, mbpp_leaks, humaneval_leaks) -- the same triple the
    build-time gate reports, computed from the artifact instead of the builder."""
    g, m, e = [], [], []
    for r in rows:
        rid = r.get("id", "")
        meta = r.get("meta") or {}
        if rid.startswith("gpqa_main/"):
            if rid.split("/", 1)[1] in h["gpqa_ids"]:
                g.append(rid)
            elif meta.get("question_sha256") in h["gpqa_hashes"]:
                g.append(f"{rid}[hash]")
        elif rid.startswith("mbpp/"):
            tail = rid.split("/", 1)[1]
            if tail.isdigit() and int(tail) in h["mbpp_ids"]:
                m.append(rid)
            if meta.get("code_sha256") in h["he_hashes"]:
                e.append(rid)
    return g, m, e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    a = ap.parse_args()

    p = pathlib.Path(a.pool)
    if not p.is_file():
        sys.exit(f"REFUSE: no pool at {p}")
    rows = [json.loads(x) for x in p.open() if x.strip()]
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if not rows:
        sys.exit(f"REFUSE: pool {p} is empty")

    h = load_holdouts()
    print(f"pool     : {p}")
    print(f"sha256   : {sha[:32]}  rows={len(rows)}")
    print(f"holdouts : diamond={h['n']['diamond']} mbpp_test={h['n']['mbpp_test']} "
          f"humaneval={h['n']['humaneval']}")

    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not ok:
            fails += 1

    print("\n=== control: each gate must FIRE on a real holdout row ===")
    poison_g = {"id": f"gpqa_main/{sorted(h['gpqa_ids'])[0]}", "meta": {}}
    poison_m = {"id": f"mbpp/{sorted(h['mbpp_ids'])[0]}", "meta": {}}
    poison_e = {"id": "mbpp/999999", "meta": {"code_sha256": sorted(h["he_hashes"])[0]}}
    cg, cm, ce = leaks([poison_g], h)
    check("GPQA gate fires on a Diamond id", len(cg) == 1, str(cg))
    cg2, _, _ = leaks([{"id": "gpqa_main/NOT_AN_ID",
                        "meta": {"question_sha256": sorted(h["gpqa_hashes"])[0]}}], h)
    check("GPQA gate fires on a Diamond question hash", len(cg2) == 1, str(cg2))
    _, cm, _ = leaks([poison_m], h)
    check("MBPP gate fires on a scored/fewshot id", len(cm) == 1, str(cm))
    _, _, ce = leaks([poison_e], h)
    check("HumanEval gate fires on a canonical solution", len(ce) == 1, str(ce))

    print("\n=== the shipped pool ===")
    g, m, e = leaks(rows, h)
    check("no GPQA Diamond row in pool", not g, f"leaks={len(g)} {g[:3]}")
    check("no MBPP scored/fewshot row in pool", not m, f"leaks={len(m)} {m[:3]}")
    check("no HumanEval solution in pool", not e, f"leaks={len(e)} {e[:3]}")

    print("\n=== every shipped row is trainable as replay ===")
    kinds = {}
    for r in rows:
        kinds[(r.get("meta") or {}).get("reward_kind", "MISSING")] = \
            kinds.get((r.get("meta") or {}).get("reward_kind", "MISSING"), 0) + 1
    check("all reward_kinds known", set(kinds) <= KNOWN_KINDS, str(kinds))
    # A replay row with length pressure is not defending capability -- it is adding to
    # the brevity signal that already cost -12.99pp on LCB in run3. An LCB row WITHOUT
    # it is inert for the thing the run is trying to measure. Both directions fail.
    bad_lam = [r["id"] for r in rows
               if (float((r.get("meta") or {}).get("length_lambda", 0.0)) != 0.0)
               != ((r.get("meta") or {}).get("reward_kind") == "lcb_exec")]
    check("length pressure is on lcb_exec and ONLY on lcb_exec", not bad_lam,
          f"{len(bad_lam)} offending rows{(' e.g. ' + str(bad_lam[:3])) if bad_lam else ''}")
    mc = [r for r in rows if (r.get("meta") or {}).get("reward_kind") == "mc_letter"]
    mb = [r for r in rows if (r.get("meta") or {}).get("reward_kind") == "mbpp_exec"]
    check("mc_letter rows carry a gold letter",
          all(str(r.get("gold") or "") in list("ABCD") for r in mc), f"{len(mc)} rows")
    check("mbpp_exec rows carry >=1 test and a reference",
          all(r["meta"].get("n_tests", 0) >= 1 and r["meta"].get("reference_code")
              for r in mb), f"{len(mb)} rows")
    check("both tiers are populated", len(mc) > 0 and len(mb) > 0,
          f"mc_letter={len(mc)} mbpp_exec={len(mb)}")

    print(f"\n{'ARTIFACT_GATE_OK' if fails == 0 else f'ARTIFACT_GATE_FAIL ({fails})'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
