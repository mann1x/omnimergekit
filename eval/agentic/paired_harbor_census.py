#!/usr/bin/env python3
"""Paired per-task census of two harbor job dirs (the TB30 A/B).

Joins on task_name and reports ONLY the intersection, so an arm that ran ahead
or crashed early cannot shift the comparison by composition. Mirrors
paired_context_census.py (the inspect/BFCL equivalent) so the two cells are
read the same way.

SCORING follows the harbor rules verified on a real run 2026-09-13:
  - job file  = <jobs_dir>/<arm>/result.json   (identified by .stats, not name)
  - trial file= <trial_dir>/result.json        (SINGULAR; the docstring lies)
  - pass/fail = verifier_result.rewards, exactly one key, value in {0,1}
  - a multi-key or non-binary reward is a REFUSAL, never a 0.0
  - the job-level pass_at_k is frequently {} even on a perfect run and is NOT
    used as the source of truth.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from loop_detect import detect
from paired_context_census import _mcnemar, _q


def _trial_dirs(job_dir: Path):
    for d in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        f = d / "result.json"
        if f.exists():
            yield d, f


def _load_arm(job_dir: Path, runaway_chars: int):
    out, refused = {}, []
    for d, f in _trial_dirs(job_dir):
        try:
            t = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(t.get("stats"), dict):       # that's the job file
            continue
        name = t.get("task_name") or d.name
        rw = ((t.get("verifier_result") or {}).get("rewards")) or {}
        if len(rw) != 1:
            refused.append((name, f"{len(rw)} reward keys")); continue
        v = next(iter(rw.values()))
        if not isinstance(v, (int, float)) or v not in (0, 1):
            refused.append((name, f"non-binary reward {v!r}")); continue

        rep = run = away = sent = turns = 0
        lens, worst = [], 0
        tf = d / "agent" / "trajectory.json"
        if tf.exists():
            try:
                traj = json.loads(tf.read_text())
            except Exception:
                traj = None
            if traj is not None:
                for txt in harbor_reasoning(traj):
                    det = detect(txt, runaway_chars)
                    turns += 1; lens.append(det["chars"])
                    rep += det["rep"]; run += det["runchar"]
                    away += det["runaway"]; sent += det["sentence"]
                    worst = max(worst, det["sentence_reps"])
        out[name] = {
            "pass": int(v), "rep": rep, "run": run, "away": away, "sent": sent,
            "worst_sent": worst, "turns": turns,
            "p50_len": int(st.median(lens)) if lens else 0,
            "chars": sum(lens),
            "looped": int(rep > 0 or run > 0 or away > 0 or sent > 0),
            "has_traj": tf.exists(),
            "exception": bool(t.get("exception_info")),
        }
    return out, refused


def _walk(obj, _d=0):
    if _d > 12:
        return
    if isinstance(obj, dict):
        for k in ("reasoning", "reasoning_content", "thinking"):
            v = obj.get(k)
            if isinstance(v, str) and v:
                yield v
        if obj.get("role") in ("assistant", "model") or "response" in obj:
            v = obj.get("content", obj.get("response"))
            if isinstance(v, str) and v:
                yield v
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, dict) and isinstance(it.get("text"), str):
                        yield it["text"]
        for v in obj.values():
            if isinstance(v, (dict, list)):
                yield from _walk(v, _d + 1)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, (dict, list)):
                yield from _walk(it, _d + 1)


def harbor_reasoning(traj):
    """Yield ONLY the model's reasoning channel from a terminus-2 trajectory.

    Structure (verified 2026-09-13): {"steps":[{"reasoning_content": str,
    "message": ..., "observation": ...}]}. We scan `reasoning_content` ONLY.

    WHY NOT `message`/`observation`: the re-injection hypothesis is about the
    THINKING channel looping. `message` holds the agent's actions — an agent
    that rewrites the same file three times legitimately repeats its own code
    lines, and `observation` is raw terminal output. Scanning them made the
    sentence detector fire on `sample_envelope <- function(...)` x3 and the
    char detector fire on indentation. Those are not loops. Measuring the
    channel under test is both cleaner and the actual hypothesis.
    """
    if isinstance(traj, dict):
        steps = traj.get("steps")
        if isinstance(steps, list):
            for st_ in steps:
                if not isinstance(st_, dict):
                    continue
                r = st_.get("reasoning_content") or st_.get("reasoning")
                if isinstance(r, str) and r.strip():
                    yield r
            return
    # unknown shape: yield nothing rather than guess. A loop_rate over zero
    # parsed blocks is WITHHELD by the caller, never reported as 0.0.
    return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", required=True, help="job dir, e.g. .../tb30/armA")
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--label-a", default="armA")
    ap.add_argument("--label-b", default="armB")
    ap.add_argument("--runaway-chars", type=int, default=10000)
    ap.add_argument("--json-out")
    a = ap.parse_args()

    A, ra = _load_arm(Path(a.arm_a), a.runaway_chars)
    B, rb = _load_arm(Path(a.arm_b), a.runaway_chars)
    shared = sorted(set(A) & set(B))
    print(f"=== PAIRED HARBOR CENSUS  {a.label_a} vs {a.label_b} ===")
    print(f"trials: {a.label_a}={len(A)}  {a.label_b}={len(B)}  SHARED={len(shared)}")
    if ra or rb:
        print(f"REFUSED (unscoreable rewards): {a.label_a}={ra} {a.label_b}={rb}")
    if not shared:
        print("FAIL: empty intersection"); return 1
    only = sorted((set(A) ^ set(B)))
    if only:
        print(f"NOTE: {len(only)} task(s) in only one arm, EXCLUDED: {only}")

    nt_a = sum(1 for i in shared if not A[i]["has_traj"])
    nt_b = sum(1 for i in shared if not B[i]["has_traj"])
    if nt_a or nt_b:
        print(f"WARN: missing trajectory.json — {a.label_a}:{nt_a} {a.label_b}:{nt_b}"
              " (their loop flags are absence, not evidence)")

    pa = sum(A[i]["pass"] for i in shared); pb = sum(B[i]["pass"] for i in shared)
    b_ = sum(1 for i in shared if A[i]["pass"] and not B[i]["pass"])
    c_ = sum(1 for i in shared if B[i]["pass"] and not A[i]["pass"])
    print("\n-- task success (paired, McNemar exact) --")
    print(f"   {a.label_a}={pa}/{len(shared)} ({100*pa/len(shared):.1f}%)   "
          f"{a.label_b}={pb}/{len(shared)} ({100*pb/len(shared):.1f}%)")
    print(f"   discordant {b_}/{c_}  p={_mcnemar(b_, c_):.4g}"
          f"{'   UNDERPOWERED (<10 discordant)' if b_+c_ < 10 else ''}")

    print("\n-- LOOPING (the endpoint) --")
    for key, lab in (("looped", "any detector"), ("sent", "SENTENCE only")):
        la = sum(1 for i in shared if A[i][key]); lb = sum(1 for i in shared if B[i][key])
        bb = sum(1 for i in shared if A[i][key] and not B[i][key])
        cc = sum(1 for i in shared if B[i][key] and not A[i][key])
        print(f"   {lab:14s}: {a.label_a}={la}/{len(shared)}  {a.label_b}={lb}/{len(shared)}"
              f"  discordant {bb}/{cc}  p={_mcnemar(bb, cc):.4g}"
              f"{'  UNDERPOWERED' if bb+cc < 10 else ''}")
    print(f"   worst in-block sentence repetition  {a.label_a}="
          f"{max(A[i]['worst_sent'] for i in shared)}x  {a.label_b}="
          f"{max(B[i]['worst_sent'] for i in shared)}x")
    print(f"   rep/runchar/runaway/SENT turns  {a.label_a}: "
          f"{sum(A[i]['rep'] for i in shared)}/{sum(A[i]['run'] for i in shared)}/"
          f"{sum(A[i]['away'] for i in shared)}/{sum(A[i]['sent'] for i in shared)}"
          f"   {a.label_b}: {sum(B[i]['rep'] for i in shared)}/"
          f"{sum(B[i]['run'] for i in shared)}/{sum(B[i]['away'] for i in shared)}/"
          f"{sum(B[i]['sent'] for i in shared)}")

    print("\n-- agent turns / output volume --")
    for key, lab in (("turns", "assistant turns"), ("chars", "total chars"),
                     ("p50_len", "per-block p50 chars")):
        va = [A[i][key] for i in shared]; vb = [B[i][key] for i in shared]
        print(f"   {lab:20s} {a.label_a}: p50={_q(va,.5):>8,}  "
              f"{a.label_b}: p50={_q(vb,.5):>8,}")

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(
            {"shared": len(shared), "pass_a": pa, "pass_b": pb,
             "mcnemar_pass": _mcnemar(b_, c_),
             "per_task": {i: {"a": A[i], "b": B[i]} for i in shared}},
            indent=2, default=str))
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
