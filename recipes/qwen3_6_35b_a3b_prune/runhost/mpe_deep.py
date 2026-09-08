#!/usr/bin/env python
"""Two open questions the register sweep did not settle.

Q1. armG's java = 32/100 with 0 empty and 1 think-marker. 68 failures by some OTHER
    mechanism. What is it? Print the actual completions.
Q2. armI loses rs-8 / js-11. MPE caps at max_gen_toks=1024 and summary token_stats show
    max=1023 for EVERY arm, so the cap does bind somewhere. If armI writes longer preambles
    it will hit the cap more. Measure length distribution, not just markers.

Also decompose armI-vs-armD newly-broken by a length bucket so "ran out of room" is
separated from "said nothing" and from "narrated instead of coding".
"""
import glob
import json
import os
import statistics

R = "/srv/ml/eval_results/ream_arms/multipl_e_100"
LANGS = {"humaneval-rs": "rs", "humaneval-java": "java", "humaneval-js": "js"}


def samples(arm):
    p = os.path.join(R, arm, "mpe_result.samples.jsonl")
    out = {}
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("task_id") or d.get("doc_id")
            if t:
                out[t] = d.get("completion") or ""
    return out


def statuses(arm):
    out = {}
    for ldir, lang in LANGS.items():
        for p in glob.glob(os.path.join(R, arm, "results", ldir, "*.results.json")):
            if p.endswith("_summary.json"):
                continue
            d = json.load(open(p))
            res = d.get("results") or []
            if res:
                out[(lang, d.get("name"))] = res[0].get("status", "?")
    return out


# ---------- Q1: armG java ----------
print("=" * 76)
print("Q1  armG java = 32/100. What are the 68 failures?")
Sg, STg = samples("reamG_ourssal_merge_gs2"), statuses("reamG_ourssal_merge_gs2")
bad = [(l, n) for (l, n), st in STg.items() if l == "java" and st != "OK"]
lens = sorted(len(Sg.get("java::%s" % n, "")) for _, n in bad)
print("  n_fail=%d  completion chars: min=%d p50=%d max=%d"
      % (len(bad), lens[0], statistics.median(lens), lens[-1]))
Sd = samples("reamD_ourssal_nomerge")
for _, n in bad[:4]:
    c = Sg.get("java::%s" % n, "")
    d = Sd.get("java::%s" % n, "")
    print("\n  --- %s  [%s]  armG_len=%d armD_len=%d"
          % (n, STg[("java", n)], len(c), len(d)))
    print("      armG: %r" % c[:300])
    print("      armD: %r" % d[:160])

# ---------- Q2: length distribution + delta decomposition ----------
print("\n" + "=" * 76)
print("Q2  completion-length distribution per arm (cap = 1024 tok; ~3.5-4 chars/tok)")
print("%-12s %-5s | %6s %6s %6s %6s | %s" %
      ("arm", "lang", "p50", "p90", "max", ">3000c", "near-cap share of FAILURES"))
for label, arm in (("armD", "reamD_ourssal_nomerge"),
                   ("armI p12", "hybrid_p12_ourssal_reapfloor"),
                   ("armJ p24", "hybrid_p24_ourssal_reapfloor"),
                   ("armG gs2", "reamG_ourssal_merge_gs2"),
                   ("base256e", "base256e_imat")):
    S, ST = samples(arm), statuses(arm)
    for lang in ("rs", "java", "js"):
        ks = [(l, n) for (l, n) in ST if l == lang]
        L = sorted(len(S.get("%s::%s" % (l, n), "")) for l, n in ks)
        big = sum(1 for x in L if x > 3000)
        fails = [(l, n) for (l, n) in ks if ST[(l, n)] != "OK"]
        fbig = sum(1 for l, n in fails if len(S.get("%s::%s" % (l, n), "")) > 3000)
        print("%-12s %-5s | %6d %6d %6d %6d | %d/%d fails are >3000c"
              % (label, lang, statistics.median(L), L[int(len(L) * .9)], L[-1], big,
                 fbig, len(fails)))
    print()

# ---------- armI vs armD newly-broken, bucketed ----------
print("=" * 76)
print("armI-vs-armD newly-broken, bucketed by mechanism")
Si, STi = samples("hybrid_p12_ourssal_reapfloor"), statuses("hybrid_p12_ourssal_reapfloor")
STd = statuses("reamD_ourssal_nomerge")
for lang in ("rs", "java", "js"):
    extra = [n for (l, n), st in STi.items()
             if l == lang and st != "OK" and STd.get((l, n)) == "OK"]
    b = {"empty": 0, "near_cap>3000c": 0, "short_partial<200c": 0, "mid": 0}
    for n in extra:
        c = Si.get("%s::%s" % (lang, n), "")
        if not c.strip():
            b["empty"] += 1
        elif len(c) > 3000:
            b["near_cap>3000c"] += 1
        elif len(c) < 200:
            b["short_partial<200c"] += 1
        else:
            b["mid"] += 1
    print("  %-5s newly-broken=%-3d  %s" % (lang, len(extra), b))
