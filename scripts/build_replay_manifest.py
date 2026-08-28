#!/usr/bin/env python3
"""Emit the PUBLISHABLE form of the GEPO replay pool: manifest + the MBPP tier.

WHY THE WHOLE POOL CANNOT BE COMMITTED
--------------------------------------
omnimergekit is PUBLIC. `gepo_replay_pool.jsonl` embeds 250 GPQA questions together
with their correct answers in plaintext. GPQA is a GATED dataset that ships a canary
string precisely so its questions do not end up in web crawls and leak into future
pretraining sets. Publishing them here would contaminate GPQA for everyone, not just
for us, and it is not ours to redistribute.

But "the builder is committed, trust it" leaves the repo with no record of WHAT was
actually trained on -- and a pool is an experimental basis, so an unrecorded basis is
an unverifiable result. This closes that gap without publishing gated content:

  * MBPP tier (CC-BY-4.0, already public) -> committed IN FULL. 471 rows, usable.
  * GPQA tier -> committed as IDENTITY ONLY: Record ID, gold letter, and the
    question's sha256. No question text, no answer text, no distractors. Someone
    without GPQA access learns nothing; someone WITH legitimate access can verify
    their rebuild matches ours row-for-row.

The manifest also pins the full pool's sha256, so a future run can prove it trained
on the same basis this repo describes.
[[feedback_verify_eval_basis_by_hash_before_tabulating]]

Run:  python scripts/build_replay_manifest.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
POOL = REPO / "eval" / "replay" / "gepo_replay_pool.jsonl"
MANIFEST = REPO / "eval" / "replay" / "gepo_replay_pool.MANIFEST.json"
MBPP_OUT = REPO / "eval" / "replay" / "gepo_replay_pool.mbpp.jsonl"

# Fields that may never reach a public file for the gated tier.
GATED_FIELDS = ("prompt", "choices")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    a = ap.parse_args()

    p = pathlib.Path(a.pool)
    if not p.is_file():
        sys.exit(f"REFUSE: no pool at {p} -- build it first "
                 f"(python scripts/build_gepo_replay_pool.py)")
    rows = [json.loads(x) for x in p.open() if x.strip()]
    sha = hashlib.sha256(p.read_bytes()).hexdigest()

    gpqa, mbpp = [], []
    for r in rows:
        kind = (r.get("meta") or {}).get("reward_kind")
        if kind == "mc_letter":
            gpqa.append({
                "id": r["id"],
                "gold": r["gold"],
                "question_sha256": r["meta"]["question_sha256"],
                "domain": r["meta"].get("domain", ""),
            })
        elif kind == "mbpp_exec":
            mbpp.append(r)
        else:
            sys.exit(f"REFUSE: unknown reward_kind {kind!r} on {r.get('id')}")

    # Belt: prove no gated text is about to be written.
    for e in gpqa:
        for f in GATED_FIELDS:
            if f in e:
                sys.exit(f"REFUSE: gated field {f!r} leaked into the manifest entry "
                         f"{e['id']} -- this file is published.")

    manifest = {
        "pool": p.name,
        "pool_sha256": sha,
        "rows": len(rows),
        "tiers": {"mc_letter": len(gpqa), "mbpp_exec": len(mbpp)},
        "builder": "scripts/build_gepo_replay_pool.py",
        "gate": "scripts/gate_replay_artifact.py",
        "note": ("GPQA tier is listed by identity only (Record ID + gold letter + "
                 "question sha256). GPQA is gated; its question and answer text is "
                 "deliberately absent. Rebuild with the committed builder and a "
                 "legitimate GPQA copy to obtain the full pool byte-for-byte."),
        "mc_letter_index": sorted(gpqa, key=lambda e: e["id"]),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    MBPP_OUT.write_text("".join(json.dumps(r) + "\n" for r in mbpp))
    print(f"pool sha256      : {sha[:32]}  rows={len(rows)}")
    print(f"manifest         : {MANIFEST.relative_to(REPO)}  "
          f"({len(gpqa)} gpqa identities, no gated text)")
    print(f"mbpp tier (full) : {MBPP_OUT.relative_to(REPO)}  ({len(mbpp)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
