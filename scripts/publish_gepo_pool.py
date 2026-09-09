#!/usr/bin/env python3
"""Publish a pool file to the ManniX-ITA/gepo-pools dataset, with its census.

A pool row's `length_lambda`, `think` and `length_budget` are the three fields that
decide what a run actually trains, and none of them is visible from the filename. So
this writes a MANIFEST beside the file and an INDEX entry, both carrying the FULL
per-tier census -- rows, reward_kind, think, lambda AND budget -- and refuses to
publish a pool whose tiers are internally inconsistent.

`SHA256SUMS` is regenerated from what is actually uploaded, never hand-edited.
[[feedback_unrecorded_axis_filename_is_not_provenance]]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys


def census(rows: list[dict]) -> dict:
    def count(fn):
        return dict(collections.Counter(str(fn(r)) for r in rows))

    tiers: dict[str, dict] = {}
    for r in rows:
        m = r.get("meta") or {}
        k = f"{m.get('reward_kind')}/{'T' if m.get('think') else 'N'}"
        t = tiers.setdefault(k, {"rows": 0, "length_lambda": set(),
                                 "length_budget": set(), "sources": collections.Counter()})
        t["rows"] += 1
        t["length_lambda"].add(m.get("length_lambda"))
        t["length_budget"].add(m.get("length_budget"))
        t["sources"][r.get("source", "?")] += 1

    out_tiers = {}
    for k, t in sorted(tiers.items()):
        lams, buds = sorted(t["length_lambda"], key=str), sorted(t["length_budget"], key=str)
        if len(lams) != 1:
            sys.exit(f"REFUSE: tier {k} carries MIXED length_lambda {lams}. A tier is "
                     "the unit a budget and a lambda are chosen for; two values in one "
                     "tier means the census cannot describe what was trained.")
        if len(buds) != 1:
            sys.exit(f"REFUSE: tier {k} carries MIXED length_budget {buds}.")
        lam, bud = lams[0], buds[0]
        if lam is not None and float(lam or 0) > 0 and not bud:
            sys.exit(f"REFUSE: tier {k} has length_lambda={lam} but no length_budget. "
                     "Publishing it would ship a driver tier that trains as pure "
                     "correctness.")
        out_tiers[k] = {"rows": t["rows"], "length_lambda": lam, "length_budget": bud,
                        "role": "driver" if float(lam or 0) > 0 else "replay",
                        "sources": dict(t["sources"])}

    n_driver = sum(v["rows"] for v in out_tiers.values() if v["role"] == "driver")
    return {
        "rows": len(rows),
        "sources": count(lambda r: r.get("source")),
        "reward_kind": count(lambda r: (r.get("meta") or {}).get("reward_kind")),
        "think": count(lambda r: bool((r.get("meta") or {}).get("think"))),
        "length_lambda": count(lambda r: (r.get("meta") or {}).get("length_lambda")),
        "length_budget": count(lambda r: (r.get("meta") or {}).get("length_budget")),
        "tiers": out_tiers,
        "driver_rows": n_driver,
        "driver_share": round(100.0 * n_driver / max(len(rows), 1), 1),
    }


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--path-in-repo", required=True,
                    help="e.g. source/driver/efficiency-v3-248.jsonl")
    ap.add_argument("--repo", default="ManniX-ITA/gepo-pools")
    ap.add_argument("--builder", default="scripts/build_gepo_mixed_pool.py")
    ap.add_argument("--notes", default="")
    ap.add_argument("--measured", default="",
                    help="smoke measured.json, embedded into the manifest so the "
                         "budgets can be traced to the run that produced them.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = pathlib.Path(a.pool)
    if not src.is_file():
        sys.exit(f"REFUSE: missing pool {src}")
    rows = [json.loads(x) for x in src.open() if x.strip()]
    c = census(rows)

    manifest = {
        "file": a.path_in_repo,
        "sha256": sha256_file(src),
        "builder": a.builder,
        "census": c,
        "notes": a.notes,
    }
    if a.measured:
        mp = pathlib.Path(a.measured)
        if not mp.is_file():
            sys.exit(f"REFUSE: --measured {mp} does not exist.")
        manifest["measured"] = json.loads(mp.read_text())

    print(json.dumps({"rows": c["rows"], "driver_share": c["driver_share"],
                      "tiers": c["tiers"]}, indent=2))
    if a.dry_run:
        print("\n[dry-run] nothing uploaded")
        return 0

    from huggingface_hub import HfApi, hf_hub_download

    token = __import__("os").environ.get("HF_TOKEN")
    if not token:
        sys.exit("REFUSE: HF_TOKEN must be exported.")
    api = HfApi(token=token)

    man_path = a.path_in_repo.rsplit(".jsonl", 1)[0] + ".MANIFEST.json"
    tmp = src.with_suffix(".MANIFEST.json")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")

    api.upload_file(path_or_fileobj=str(src), path_in_repo=a.path_in_repo,
                    repo_id=a.repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(tmp), path_in_repo=man_path,
                    repo_id=a.repo, repo_type="dataset")
    print(f"uploaded {a.path_in_repo} + {man_path}")

    # INDEX.json -- read the LIVE copy, add/replace this entry, write it back.
    idx_local = hf_hub_download(a.repo, "INDEX.json", repo_type="dataset")
    idx = json.loads(pathlib.Path(idx_local).read_text())
    idx[a.path_in_repo] = {"rows": c["rows"], "sha256": manifest["sha256"],
                           **{k: c[k] for k in ("sources", "length_lambda",
                                                "reward_kind", "think", "tiers",
                                                "driver_share")},
                           "builder": a.builder}
    if a.measured:
        idx[a.path_in_repo]["measured"] = manifest["measured"]
    idx_tmp = src.parent / "INDEX.json"
    idx_tmp.write_text(json.dumps(idx, indent=2, sort_keys=True) + "\n")
    api.upload_file(path_or_fileobj=str(idx_tmp), path_in_repo="INDEX.json",
                    repo_id=a.repo, repo_type="dataset")
    print("updated INDEX.json")

    # SHA256SUMS -- regenerate from the repo listing so it can never drift from what
    # is actually there.
    files = sorted(f for f in api.list_repo_files(a.repo, repo_type="dataset")
                   if f != "SHA256SUMS" and not f.startswith("."))
    lines = []
    for f in files:
        lp = hf_hub_download(a.repo, f, repo_type="dataset")
        lines.append(f"{sha256_file(pathlib.Path(lp))}  ./{f}")
    sums = src.parent / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n")
    api.upload_file(path_or_fileobj=str(sums), path_in_repo="SHA256SUMS",
                    repo_id=a.repo, repo_type="dataset")
    print(f"regenerated SHA256SUMS ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
