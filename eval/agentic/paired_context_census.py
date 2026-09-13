#!/usr/bin/env python3
"""Paired per-task census of two inspect .eval arms.

WHY PAIRED. The arms run the same task list but finish at different rates, so
an unpaired distribution over "whatever each arm has done so far" compares
different task sets — a composition artefact, not an arm effect. This joins on
task id and reports ONLY the intersection, so composition is balanced by
construction. It states the intersection size and what was dropped.

METRICS
  ctx_tokens  = input_tokens + input_tokens_cache_read, summed over every model
                call in the sample. This is the PROMPT cost the template drives
                (llama.cpp's `stop processing: n_tokens` is the same quantity
                per call). Cache-read tokens are still context the model must
                carry — excluding them would understate re-injection.
  out_tokens  = output_tokens (fresh generation).
  score       = the bfcl scorer's binary value.

STATS
  accuracy    -> McNemar exact on discordant pairs (paired, binary).
  token ratios-> Wilcoxon signed-rank on per-task paired differences, plus the
                 median per-task ratio. A ratio of MEDIANS is not a median of
                 RATIOS; the paired quantity is the latter.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

# inspect_ai is imported LAZILY inside _load(): paired_harbor_census.py reuses
# _mcnemar/_q from this module and reads harbor job dirs, which have nothing to
# do with inspect .eval logs. A top-level import made the harbor census
# unrunnable on any interpreter without the BFCL venv (bs2, 2026-09-13).
from loop_detect import detect, message_texts  # THE detector — never redeclare


def _usage(s):
    ctx = out = 0
    for u in (s.model_usage or {}).values():
        ctx += (u.input_tokens or 0) + (u.input_tokens_cache_read or 0)
        out += u.output_tokens or 0
    return ctx, out


def _score(s):
    for v in (s.scores or {}).values():
        x = v.value
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int, float)):
            return int(x)
        if isinstance(x, str):
            return 1 if x.upper() in ("C", "CORRECT", "TRUE", "1") else 0
    return None


def _loop(s, runaway_chars):
    """Per-sample degeneration over assistant text + reasoning, via the shared
    detector. Sentence-level repetition is counted SEPARATELY from the char
    detectors: on real data it fires 10-30x more often, and a census without it
    reports a null that is an artefact of the instrument."""
    rep = run = away = sent = turns = 0
    lens = []
    max_sent = 0
    for m in s.messages or []:
        if getattr(m, "role", None) != "assistant":
            continue
        for t in message_texts(m):
            d = detect(t, runaway_chars)
            turns += 1
            lens.append(d["chars"])
            rep += d["rep"]
            run += d["runchar"]
            away += d["runaway"]
            sent += d["sentence"]
            max_sent = max(max_sent, d["sentence_reps"])
    return rep, run, away, sent, max_sent, turns, lens


def _load(path, runaway_chars):
    from inspect_ai.log import read_eval_log  # lazy: see note at the imports

    log = read_eval_log(str(path))
    out = {}
    for s in log.samples or []:
        ctx, gen = _usage(s)
        rep, run, away, sent, max_sent, turns, lens = _loop(s, runaway_chars)
        out[str(s.id)] = {
            "score": _score(s), "ctx": ctx, "out": gen,
            # WALL TIME CAVEAT: total_time is client-side and includes any
            # contention from a concurrently-running sibling arm. Paired per
            # task it is still the best available comparison, but it is NOT a
            # contention-free benchmark -- for that, run one arm at a time.
            "secs": float(getattr(s, "working_time", 0) or
                          getattr(s, "total_time", 0) or 0),
            "rep": rep, "run": run, "away": away, "sent": sent,
            "max_sent": max_sent, "turns": turns,
            "p50_len": int(st.median(lens)) if lens else 0,
            "looped": int(rep > 0 or run > 0 or away > 0 or sent > 0),
            "msgs": len(s.messages or []),
        }
    return log.status, out


def _mcnemar(b, c):
    """Exact two-sided binomial on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank, normal approx with tie correction.
    Returns (p, n_nonzero). Not valid below ~10 non-zero pairs — caller gates."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 10:
        return None, n
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    wp = sum(ranks[i] for i in range(n) if d[i] > 0)
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sd == 0:
        return None, n
    z = (wp - mu) / sd
    return math.erfc(abs(z) / math.sqrt(2)), n


def _q(v, p):
    if not v:
        return 0
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", required=True)
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--label-a", default="armA")
    ap.add_argument("--label-b", default="armB")
    ap.add_argument("--runaway-chars", type=int, default=10000)
    ap.add_argument("--only-ids", help=(
        "JSON list of task ids to restrict the comparison to. Use when part of "
        "a run is known to be on a different basis -- e.g. samples in flight "
        "when a server died accumulate retry storms (one composite sample "
        "reached 54,199,572 context tokens on 2026-09-12 and wrecked every "
        "mean it entered). Excluding them is a BASIS decision and must be "
        "stated, never silent."))
    ap.add_argument("--json-out")
    a = ap.parse_args()

    sa, A = _load(a.arm_a, a.runaway_chars)
    sb, B = _load(a.arm_b, a.runaway_chars)
    shared = sorted(set(A) & set(B))
    if a.only_ids:
        keep = set(json.loads(Path(a.only_ids).read_text()))
        dropped = [i for i in shared if i not in keep]
        shared = [i for i in shared if i in keep]
        print(f"RESTRICTED to --only-ids {a.only_ids}: kept {len(shared)}, "
              f"dropped {len(dropped)} off-basis task(s)")
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))

    print(f"=== PAIRED CENSUS  {a.label_a} vs {a.label_b} ===")
    print(f"status: {a.label_a}={sa}  {a.label_b}={sb}")
    print(f"samples: {a.label_a}={len(A)}  {a.label_b}={len(B)}  "
          f"SHARED={len(shared)}  only_{a.label_a}={len(only_a)}  "
          f"only_{a.label_b}={len(only_b)}")
    if not shared:
        print("FAIL: empty intersection; nothing comparable")
        return 1
    if sa == "started" or sb == "started":
        print("NOTE: at least one arm is STILL RUNNING — this is an interim, "
              "balanced-composition reading over the shared tasks only.")

    rows = {}
    for key, lab in (("ctx", "prompt/context tokens"),
                     ("out", "output tokens"),
                     ("secs", "WALL SECONDS / task (contention-confounded)")):
        va = [A[i][key] for i in shared]
        vb = [B[i][key] for i in shared]
        ratios = [A[i][key] / B[i][key] for i in shared if B[i][key] > 0]
        p, nz = _wilcoxon([A[i][key] - B[i][key] for i in shared])
        print(f"\n-- {lab} (per task, summed over calls) --")
        print(f"   {a.label_a}: p50={_q(va,.5):>9,.0f}  p90={_q(va,.9):>9,.0f}  "
              f"mean={st.mean(va):>9,.0f}  sum={sum(va):>12,.0f}")
        print(f"   {a.label_b}: p50={_q(vb,.5):>9,.0f}  p90={_q(vb,.9):>9,.0f}  "
              f"mean={st.mean(vb):>9,.0f}  sum={sum(vb):>12,.0f}")
        mr = st.median(ratios) if ratios else float("nan")
        print(f"   median per-task RATIO {a.label_a}/{a.label_b} = {mr:.3f}"
              f"   (total ratio {sum(va)/max(1,sum(vb)):.3f})")
        print(f"   Wilcoxon signed-rank p = "
              f"{('%.4g' % p) if p is not None else 'n/a'}  (n_nonzero={nz}"
              f"{'; UNDERPOWERED <10' if nz < 10 else ''})")
        rows[key] = {"p50_a": _q(va, .5), "p50_b": _q(vb, .5),
                     "p90_a": _q(va, .9), "p90_b": _q(vb, .9),
                     "median_ratio": mr, "wilcoxon_p": p, "n_nonzero": nz}

    # accuracy (paired, binary)
    pa = [A[i]["score"] for i in shared]
    pb = [B[i]["score"] for i in shared]
    if any(x is None for x in pa + pb):
        print("\n-- accuracy -- WITHHELD: a shared sample has no score")
    else:
        b_ = sum(1 for i in range(len(shared)) if pa[i] == 1 and pb[i] == 0)
        c_ = sum(1 for i in range(len(shared)) if pa[i] == 0 and pb[i] == 1)
        p = _mcnemar(b_, c_)
        print("\n-- accuracy (paired, McNemar exact) --")
        print(f"   {a.label_a}={sum(pa)}/{len(pa)} ({100*sum(pa)/len(pa):.1f}%)"
              f"   {a.label_b}={sum(pb)}/{len(pb)} ({100*sum(pb)/len(pb):.1f}%)")
        print(f"   discordant: {a.label_a}-only={b_}  {a.label_b}-only={c_}"
              f"   p = {p:.4g}"
              f"{'   UNDERPOWERED (<10 discordant)' if b_+c_ < 10 else ''}")
        rows["accuracy"] = {"a": sum(pa), "b": sum(pb), "n": len(pa),
                            "disc_a": b_, "disc_b": c_, "mcnemar_p": p}

    # degeneration endpoint
    la = sum(A[i]["looped"] for i in shared)
    lb = sum(B[i]["looped"] for i in shared)
    b_ = sum(1 for i in shared if A[i]["looped"] and not B[i]["looped"])
    c_ = sum(1 for i in shared if B[i]["looped"] and not A[i]["looped"])
    print(f"\n-- loop census (THE endpoint; runaway>={a.runaway_chars} chars) --")
    print(f"   {a.label_a}: {la}/{len(shared)} looped   "
          f"{a.label_b}: {lb}/{len(shared)} looped")
    print(f"   rep/runchar/runaway/SENTENCE turns  {a.label_a}: "
          f"{sum(A[i]['rep'] for i in shared)}/"
          f"{sum(A[i]['run'] for i in shared)}/"
          f"{sum(A[i]['away'] for i in shared)}/"
          f"{sum(A[i]['sent'] for i in shared)}   {a.label_b}: "
          f"{sum(B[i]['rep'] for i in shared)}/"
          f"{sum(B[i]['run'] for i in shared)}/"
          f"{sum(B[i]['away'] for i in shared)}/"
          f"{sum(B[i]['sent'] for i in shared)}")
    print(f"   worst in-block sentence repetition  {a.label_a}="
          f"{max(A[i]['max_sent'] for i in shared)}x  {a.label_b}="
          f"{max(B[i]['max_sent'] for i in shared)}x")
    # sentence-only loop rate: the endpoint with the length proxy removed
    sa_ = sum(1 for i in shared if A[i]["sent"])
    sb_ = sum(1 for i in shared if B[i]["sent"])
    bb = sum(1 for i in shared if A[i]["sent"] and not B[i]["sent"])
    cc = sum(1 for i in shared if B[i]["sent"] and not A[i]["sent"])
    print(f"   SENTENCE-loop only: {a.label_a}={sa_}/{len(shared)}  "
          f"{a.label_b}={sb_}/{len(shared)}  discordant {bb}/{cc}  "
          f"McNemar p = {_mcnemar(bb, cc):.4g}"
          f"{'   UNDERPOWERED (<10 discordant)' if bb+cc < 10 else ''}")
    print(f"   discordant: {b_}/{c_}  McNemar p = {_mcnemar(b_, c_):.4g}"
          f"{'   UNDERPOWERED (<10 discordant)' if b_+c_ < 10 else ''}")
    rows["loop"] = {"a": la, "b": lb, "n": len(shared),
                    "disc_a": b_, "disc_b": c_, "mcnemar_p": _mcnemar(b_, c_)}

    print("\n-- assistant turns / sample --")
    print(f"   {a.label_a}: p50={_q([A[i]['turns'] for i in shared],.5)}"
          f"  {a.label_b}: p50={_q([B[i]['turns'] for i in shared],.5)}")
    print(f"   reasoning+content p50 chars  {a.label_a}="
          f"{_q([A[i]['p50_len'] for i in shared],.5)}  {a.label_b}="
          f"{_q([B[i]['p50_len'] for i in shared],.5)}")

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(
            {"shared": len(shared), "status_a": sa, "status_b": sb,
             "interim": sa == "started" or sb == "started", **rows},
            indent=2, default=str))
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
