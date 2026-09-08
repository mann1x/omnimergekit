#!/usr/bin/env python
"""LCB-v6-77q two-axis readout: pass rate AND the length/loop/rumination profile.

The winner is not the highest pass@1. A model that scores well by writing 30k tokens of
circular reasoning before landing the answer is not usable, and neither HE+ nor MPE-100 can
see that -- both are short-completion benches (MPE p50 ~250 chars), so a ruminating model
looks identical to a clean one there. LCB-v6-77q is all-hard with a 32k generation budget,
which is precisely where rumination becomes visible.

AXIS 1 (score): summary.json .score ONLY -- omk_eval already picked metric+filter.
AXIS 2 (length/loop):
  * length distribution p50/p90/max over completion chars (and tokens when recorded),
  * CAP-HIT rate: completions that ran to the 32k wall. The template header records that on
    the first 16k run every cap-hit was a GENUINE cut-off (code still being written,
    repetition fraction <=0.046, ZERO degenerate loops) -- so a cap hit is NOT by itself
    rumination, and the two must be counted separately or the metric lies.
  * LOOP rate via detect_loop() from scripts/audit_full_bench.py -- the SAME detector used
    for every loop finding in this project. Importing it rather than re-deriving one keeps
    this column comparable to the 48-seed agentic gate and the v7/v8 loop tables.
  * loop x pass cross-tab, because "loops but still passes" and "loops and fails" are
    different products.

Reads only what is on disk; no GPU, no re-run.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from audit_full_bench import detect_loop  # noqa: E402

RES = os.environ.get("LCB_RES", "/srv/ml/eval_results")


def find(col):
    """col may be 'suite/name' or an absolute dir."""
    if os.path.isdir(col):
        return col
    for suite in ("qwen_suite", "ream_arms"):
        p = f"{RES}/{suite}/lcb_v6_77q/{col}"
        if os.path.isdir(p):
            return p
    if "/" in col:
        suite, name = col.split("/", 1)
        p = f"{RES}/{suite}/lcb_v6_77q/{name}"
        if os.path.isdir(p):
            return p
    return None


def text_of(r):
    """Pull the model completion out of a sample row, whatever the key is."""
    for k in ("resps", "filtered_resps", "completion", "generation", "response",
              "output", "code", "raw_response"):
        v = r.get(k)
        while isinstance(v, list) and v:
            v = v[0]
        if isinstance(v, str) and v.strip():
            return v
    return ""


def passed(r):
    for k in ("pass", "passed", "correct", "pass@1", "graded", "is_correct"):
        if k in r:
            v = r[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return v >= 0.5
    return None


def finish(r):
    for k in ("finish_reason", "stop_reason", "finish"):
        if k in r and isinstance(r[k], str):
            return r[k]
    return None


def report(col):
    d = find(col)
    if not d:
        print(f"{col:34s}  -- no result dir")
        return None
    score = None
    sp = f"{d}/summary.json"
    if os.path.exists(sp):
        s = json.load(open(sp))
        score = s.get("score")
        sampler = (s.get("sampler") or {}).get("name")
    else:
        sampler = None

    jl = f"{d}/lcb_result.samples.jsonl"
    if not os.path.exists(jl):
        print(f"{col:34s}  score={score}  (no samples.jsonl)")
        return None
    rows = []
    for line in open(jl):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

    lens, loops, caps, npass, nloop_pass, nloop_fail = [], 0, 0, 0, 0, 0
    have_pass = False
    for r in rows:
        t = text_of(r)
        lens.append(len(t))
        lp = detect_loop(t)
        if lp:
            loops += 1
        fr = finish(r)
        if fr and fr.lower() in ("length", "max_tokens", "max_new_tokens"):
            caps += 1
        p = passed(r)
        if p is not None:
            have_pass = True
            if p:
                npass += 1
            if lp:
                nloop_pass += 1 if p else 0
                nloop_fail += 0 if p else 1
    lens.sort()

    def q(f):
        return lens[min(len(lens) - 1, int(len(lens) * f))] if lens else 0

    n = len(rows)
    print(f"{col:34s} score={score if score is None else round(score,4)} "
          f"sampler={sampler} n={n}")
    # Prefer omk's own tokenized completion stats over the char proxy: tokens are what the
    # 32768 ceiling is denominated in, so max==32768 IS the cap and needs no inference.
    ts = ((json.load(open(sp)).get("token_stats") or {}).get("completion_tokens")
          if os.path.exists(sp) else None) or {}
    if ts:
        print(f"{'':34s}   TOKENS p50={ts.get('p50'):>7} p90={ts.get('p90'):>7} "
              f"max={ts.get('max'):>7}  (ceiling 32768)")
    print(f"{'':34s}   chars  p50={q(.5):7d} p90={q(.9):7d} max={lens[-1] if lens else 0:7d}")
    print(f"{'':34s}   LOOP={loops:3d} ({100*loops/max(n,1):5.1f}%)   "
          f"cap-hit={caps:3d} ({100*caps/max(n,1):5.1f}%)" +
          (f"   pass={npass}  loop&pass={nloop_pass} loop&fail={nloop_fail}"
           if have_pass else "   (no per-doc pass field)"))
    return dict(n=n, score=score, loops=loops, caps=caps, p50=q(.5), p90=q(.9))


cols = sys.argv[1:] or ["qwen256e_q6k", "qwen184e_q6k"]
print(f"{'column':34s} LCB-v6-77q: score + length/loop profile")
print("-" * 100)
for c in cols:
    report(c)
print("\nA cap-hit is a GENUINE cut-off unless LOOP also fires -- count them separately.")
print("detect_loop is scripts/audit_full_bench.py verbatim (same detector as the loop gates).")
