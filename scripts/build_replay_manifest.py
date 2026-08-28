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
# Output paths are DERIVED from --pool (see --out-prefix). They used to be constants
# pinned to the replay pool, which meant running this on a second pool would have
# overwritten the first pool's published record with the second pool's contents --
# leaving a committed manifest whose sha256 belonged to a file nobody could identify.

# Fields that may never reach a public file for the gated tier.
GATED_FIELDS = ("prompt", "choices")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--builder", default="scripts/build_gepo_replay_pool.py",
                    help="The deterministic builder that reproduces this pool. Recorded "
                         "in the manifest: it is the ONLY way a reader with legitimate "
                         "GPQA access can rebuild the withheld rows and check the "
                         "sha256, so naming the wrong one makes the manifest unusable.")
    ap.add_argument("--out-prefix", default="",
                    help="Base path for the outputs. Default: derived from --pool, so "
                         "each pool gets its own manifest instead of overwriting "
                         "another pool's.")
    a = ap.parse_args()

    p = pathlib.Path(a.pool)
    if not p.is_file():
        sys.exit(f"REFUSE: no pool at {p} -- build it first "
                 f"(python scripts/build_gepo_replay_pool.py)")
    rows = [json.loads(x) for x in p.open() if x.strip()]
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    p = p.resolve()
    prefix = pathlib.Path(a.out_prefix).resolve() if a.out_prefix else \
        p.with_suffix("")            # eval/replay/gepo_mixed_pool
    manifest_path = prefix.with_name(prefix.name + ".MANIFEST.json")
    open_path = prefix.with_name(prefix.name + ".open.jsonl")

    # PUBLISHABLE vs GATED, decided by tier and nothing else:
    #   mc_letter  -> GPQA, gated, identity only
    #   mbpp_exec  -> MBPP, CC-BY-4.0, published in full
    #   lcb_exec   -> LiveCodeBench, public contest problems, published in full
    # The mixed pool carries all three, so the older mbpp-only split would have
    # silently dropped the 128 LCB rows out of the published record -- leaving the
    # committed artifact describing a pool nobody trained on.
    PUBLISHABLE = {"mbpp_exec", "lcb_exec"}
    gpqa, openrows = [], []
    for r in rows:
        kind = (r.get("meta") or {}).get("reward_kind")
        if kind == "mc_letter":
            gpqa.append({
                "id": r["id"],
                "gold": r["gold"],
                "question_sha256": r["meta"]["question_sha256"],
                "domain": r["meta"].get("domain", ""),
                "think": r["meta"].get("think"),
            })
        elif kind in PUBLISHABLE:
            openrows.append(r)
        else:
            sys.exit(f"REFUSE: unknown reward_kind {kind!r} on {r.get('id')}")

    # Belt: prove no gated text is about to be written.
    for e in gpqa:
        for f in GATED_FIELDS:
            if f in e:
                sys.exit(f"REFUSE: gated field {f!r} leaked into the manifest entry "
                         f"{e['id']} -- this file is published.")

    tiers = {}
    for r in rows:
        k = (r.get("meta") or {}).get("reward_kind", "?")
        k += "/T" if (r.get("meta") or {}).get("think", k == "lcb_exec") else "/N"
        tiers[k] = tiers.get(k, 0) + 1
    manifest = {
        "pool": p.name,
        "pool_sha256": sha,
        "rows": len(rows),
        "tiers": dict(sorted(tiers.items())),
        "builder": a.builder,
        "gate": "scripts/gate_replay_artifact.py",
        "note": ("GPQA tier is listed by identity only (Record ID + gold letter + "
                 "question sha256). GPQA is gated; its question and answer text is "
                 "deliberately absent. Rebuild with the committed builder and a "
                 "legitimate GPQA copy to obtain the full pool byte-for-byte."),
        "mc_letter_index": sorted(gpqa, key=lambda e: e["id"]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    open_path.write_text("".join(json.dumps(r) + "\n" for r in openrows))
    print(f"pool sha256      : {sha[:32]}  rows={len(rows)}")
    print(f"tiers            : {manifest['tiers']}")
    print(f"manifest         : {manifest_path.relative_to(REPO)}  "
          f"({len(gpqa)} gpqa identities, no gated text)")
    print(f"open tiers (full): {open_path.relative_to(REPO)}  ({len(openrows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
