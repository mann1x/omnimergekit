#!/usr/bin/env python
"""Mandatory post-bench sanity check for ONE eval cell.

WHY THIS EXISTS. The rule is "always verify samples_*.jsonl after each run", and the
check must run on the RAW generations (`resps`), not the post-filter `filtered_resps`.
Reading the filtered field makes a healthy extractive bench look catastrophic: on
gsm8k_100_boxed the filtered value is the extracted NUMBER, so 192/200 responses are
under 5 characters and the median length is 2 -- which is correct behaviour, not a
truncation bug. I made exactly that misread on 2026-08-30 and nearly reported it as an
alarm. Both fields are printed here so the distinction is visible, and the VERDICT is
computed from raw only.

An anomaly means STOP and investigate before trusting the score (bug-015: chat-mode
markdown fences made HumanEval pass@1=0.0 for a reason invisible without the samples).

Usage: sanity_cell.py <results_dir>/<bench>/<served_name>
"""
import glob
import json
import statistics
import sys

d = sys.argv[1].rstrip("/")
s = json.load(open(f"{d}/summary.json"))
sm = (s.get("sampler") or {}).get("name")
ts = (s.get("token_stats") or {}).get("completion_tokens") or {}
print(f"score={s.get('score')} metric={s.get('metric')} filter={s.get('filter')} "
      f"sampler={sm} tok_sum={ts.get('sum')} tok_p50={ts.get('p50')}")

# Two naming conventions in this results tree: lm-eval writes samples_<task>_*.jsonl,
# while the multipl_e driver writes mpe_result.samples.jsonl. Globbing only the first
# made BOTH ck218 and armJ's trusted b604 comparator report "NO SAMPLES FILE -- do not
# trust this score", which is a checker that cannot see its own input.
f = sorted(glob.glob(f"{d}/**/samples_*.jsonl", recursive=True)
           + glob.glob(f"{d}/**/*.samples.jsonl", recursive=True))
if not f:
    print("VERDICT: NO SAMPLES FILE -- cannot verify, do not trust this score")
    sys.exit(2)
rows = [json.loads(l) for l in open(f[-1])]


# THREE SAMPLE FORMATS IN THIS TREE. lm-eval uses resps/filtered_resps; the LCB driver
# writes completion/cleaned; multipl_e writes its own file name. Reading only resps made
# lcb_v6_77q report "77/77 empty generations -- do not trust the score" on BOTH ck218 and
# armJ, when in fact both scored normally. A checker that misreads a format as total
# failure is worse than no checker: it trains you to ignore it.
RAW_KEYS = ("resps", "completion")
FILT_KEYS = ("filtered_resps", "cleaned")


def pick(r, key):
    keys = RAW_KEYS if key == "resps" else FILT_KEYS
    for k in keys:
        x = r.get(k)
        if x is None:
            continue
        while isinstance(x, list) and x:
            x = x[0]
        if isinstance(x, str) and x:
            return x
    return ""


raw = [pick(r, "resps") for r in rows]
filt = [pick(r, "filtered_resps") for r in rows]
n = len(rows)
empty = sum(1 for x in raw if not x.strip())
lt5 = sum(1 for x in raw if len(x.strip()) < 5)
fence = sum(1 for x in raw if "```" in x)
p50 = int(statistics.median([len(x) for x in raw])) if raw else 0
mx = max((len(x) for x in raw), key=int) if raw else 0
print(f"RAW  n={n} empty={empty} lt5={lt5} md_fence={fence} chars_p50={p50} chars_max={mx}")
print(f"FILT chars_p50={int(statistics.median([len(x) for x in filt])) if filt else 0} "
      f"(extractive benches are SUPPOSED to be tiny here -- not an anomaly)")

# Verdict on RAW only. A fenced generation is not automatically fatal (only chat-mode
# code benches scored via exec are), so it is reported as REVIEW rather than FAIL.
bad = []
if empty:
    bad.append(f"{empty} empty generations")
if lt5:
    bad.append(f"{lt5} sub-5-char generations")

# FENCES ARE NOT THE BUG-015 SIGNATURE ON THEIR OWN. Chat-mode code benches fence ~100%
# of generations by construction; what made bug-015 score pass@1=0.0 was fences reaching
# exec() with NO extraction filter. Measured 2026-08-30: humaneval_full_think has
# md_fence=164/164 on BOTH ck218 (0.945) and armJ (0.982) -- universal, stripped by
# extract_chat, and identical across arms, so it cannot explain a between-arm delta.
# The real alarm is fences together with a collapsed score or a missing extractor.
# GATE ON THE OBSERVABLE SYMPTOM, NOT ON THE FILTER NAME. First attempt keyed on
# "filter contains 'extract'", which false-positived on ifeval_100: 3 incidental fenced
# generations, filter=none, score=0.96 -> reported "bug-015 shape, do not trust". IFEval
# execs nothing and a 0.96 is proof the scorer ran. The bug-015 symptom is a COLLAPSED
# score (fences reaching exec()), so that is what this tests.
score = s.get("score") or 0
if not bad and fence:
    if score <= 0.05:
        bad.append(f"{fence} fenced generations AND score={score} is collapsed -- the "
                   f"bug-015 shape (fences reaching the scorer)")
    else:
        print(f"VERDICT: OK -- {fence}/{n} fenced, but score={score:.4f} is live, so the "
              f"scorer is not choking on them (filter={s.get('filter')})")
if bad:
    print("VERDICT: ANOMALY -- " + "; ".join(bad) + " -- STOP, do not trust the score")
    sys.exit(1)
else:
    print("VERDICT: OK")
