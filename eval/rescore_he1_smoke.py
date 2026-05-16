#!/usr/bin/env python3
"""rescore_he1_smoke.py — offline-rescore humaneval_1_smoke for completed
v5fixed sweep variants using the FIXED `extract_chat` filter.

Why offline: lm-eval cached the raw model output (`resps`) in samples_*.jsonl.
The old filter took the wrong slice and stored an unusable `filtered_resps`.
We can replay the filter on `resps` + rerun the HF `code_eval` metric — no
inference needed, takes seconds per variant.

The new filter lives in
`/shared/dev/omnimergekit/eval/lm_eval_tasks/humaneval_chat/utils_chat.py` —
this script imports `build_predictions_chat` + `pass_at_k` from there so the
rescore is bit-identical to a fresh task run.

Usage:
    HF_ALLOW_CODE_EVAL=1 python scripts/rescore_he1_smoke.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

# Tell HF code_eval we accept the risk (same env we'd set for fresh eval).
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

sys.path.insert(0, "/shared/dev/omnimergekit/eval/lm_eval_tasks/humaneval_chat")
from utils_chat import build_predictions_chat, pass_at_k  # noqa: E402


def rescore(samples_path: Path) -> dict:
    """Replay filter + scorer on a samples_*.jsonl. Returns rescored result."""
    resps_all: list[list[str]] = []
    docs_all: list[dict] = []
    refs_all: list[str] = []
    old_scores: list[float] = []
    raw_lens: list[int] = []
    with open(samples_path) as f:
        for line in f:
            d = json.loads(line)
            r = d.get("resps") or [[""]]
            # d["resps"] is list[list[str]] (outer=docs handled by jsonl line,
            # inner=repeats). We want the inner repeats list for this doc.
            resps_all.append(r[0] if r else [""])
            docs_all.append(d.get("doc") or {})
            refs_all.append(d.get("target") or "")
            old_scores.append(float(d.get("pass@1", 0.0)))
            raw_lens.append(len(r[0][0]) if r and r[0] else 0)

    # Apply new filter
    preds = build_predictions_chat(resps_all, docs_all)
    # Sanity: check what the new filter extracted
    nonempty = sum(1 for p in preds if p and p[0].strip())
    has_def = sum(1 for p in preds if p and "def " in p[0])

    # Score
    res = pass_at_k(refs_all, preds, k=[1])

    return {
        "n": len(resps_all),
        "pass_at_1_new": float(res.get("pass@1", 0.0)),
        "pass_at_1_old_avg": (sum(old_scores) / len(old_scores)) if old_scores else 0.0,
        "n_nonempty_pred": nonempty,
        "n_pred_with_def": has_def,
        "raw_len_avg": (sum(raw_lens) / len(raw_lens)) if raw_lens else 0,
    }


def main() -> int:
    root = Path(
        "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/eval_results_vllm_suite/v5fixed_sweep"
    )
    variants = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "humaneval_1_smoke").exists()
    )
    if not variants:
        print(f"FAIL: no variants with humaneval_1_smoke under {root}")
        return 1

    print(f"{'variant':<28} {'old':>6}  {'NEW':>6}  {'n':>3}  {'pred_def':>8}  {'avg_raw':>8}")
    print("-" * 74)
    summary = {}
    for v in variants:
        samp = glob.glob(
            str(root / v / "humaneval_1_smoke" / "humaneval_1_smoke" / "*nvfp4a16"
                / "lm_eval_out" / "*nvfp4a16" / "samples_humaneval_chat_*.jsonl")
        )
        if not samp:
            print(f"{v:<28} [no samples_*.jsonl]")
            continue
        r = rescore(Path(samp[0]))
        summary[v] = r
        flag = " ★" if r["pass_at_1_new"] > r["pass_at_1_old_avg"] else ""
        print(
            f"{v:<28} {r['pass_at_1_old_avg']:6.3f}  "
            f"{r['pass_at_1_new']:6.3f}{flag}  {r['n']:3d}  "
            f"{r['n_pred_with_def']:8d}  {r['raw_len_avg']:8.0f}"
        )

    # Write json for the sweep summary updater to consume
    out = Path(
        "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/logs/v5fixed_sweep_he1_rescore.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nrescore summary → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
