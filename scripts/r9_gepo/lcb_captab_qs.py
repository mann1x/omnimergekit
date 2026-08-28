"""LCB truncation cross-tab + paired both-untruncated comparison -- qwen_suite cohort.

CAP: read from each cell's own summary.json, arms that disagree are FATAL. An unequal
ceiling turns the bench into a length meter rather than a capability test. This cohort
runs 32768 (thinking 12288), NOT the 49152 of the ream_arms 48k cohort -- hardcoding
that constant reports ZERO truncations where there are real ones.

TRUNCATION SOURCE (corrected 2026-08-26): the samples file records a per-problem
`finish_reason` and server-reported `completion_tokens` (producer:
eval/lcb/lcb_llama_server.py builds rec with both). finish_reason == "length" is the
AUTHORITATIVE truncation signal and is used as such here. An earlier version of this
script re-derived truncation by re-tokenizing the completion because summary.json's
aggregate `finish_reasons` is None -- but that aggregate is merely a roll-up gap
(bug-592/#834), not missing data. The tokenizer estimate is retained only as a
CROSS-CHECK and any disagreement is printed loudly.

LENGTH CHANNEL: `completion` is the full server response; the runner logs
chars=len(completion) off the same unmodified variable, so log-chars and samples-chars
are comparable. Do NOT compare either to lm-eval `resps`, which is content-only (bug-635).
"""
import glob, json, os
from math import comb

R = "/srv/ml/eval_results/qwen_suite/lcb_v6_77q"
ARMS = [("armJ", "qwenhybridp24_q6k"), ("gepo1", "qwena3bgepo1_q6k"), ("gepo2", "qwena3bgepo2_q6k")]
TOL = 16

caps = {}
for tag, name in ARMS:
    p = os.path.join(R, name, "summary.json")
    if not os.path.exists(p):
        raise SystemExit("REFUSE: missing cell %s (%s) -- has it finished?" % (tag, p))
    caps[tag] = (json.load(open(p)).get("generation_caps") or {}).get("max_gen_toks")
if len(set(caps.values())) != 1 or None in caps.values():
    raise SystemExit("REFUSE: arms disagree on max_gen_toks: %s" % caps)
CAP = list(caps.values())[0]
print("cap read from cells: max_gen_toks=%d (trunc threshold %d)\n" % (CAP, CAP - TOL))


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def sign_test(k, n):
    """two-sided sign test on n paired non-ties, k in one direction"""
    if n == 0: return 1.0
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(name):
    p = os.path.join(R, name, "lcb_result.samples.jsonl")
    if not os.path.exists(p):
        raise SystemExit("REFUSE: no samples file at %s" % p)
    out, mism = {}, 0
    for ln in open(p):
        if not ln.strip(): continue
        d = json.loads(ln)
        tid = d.get("task_id") or d.get("doc_id")
        ntok = d.get("completion_tokens")
        fr = d.get("finish_reason")
        trunc = (fr == "length")
        if isinstance(ntok, int) and (ntok >= CAP - TOL) != trunc:
            mism += 1
        out[tid] = (ntok, trunc, bool(d.get("passed")), len(d.get("completion") or ""), fr)
    if mism:
        print("  !! %s: %d problems where finish_reason and token-count disagree "
              "-- using finish_reason (authoritative)" % (name, mism))
    return out


arms = {t: load(n) for t, n in ARMS}
print("%-7s %4s %5s %8s %6s %11s %9s %10s" % ("arm", "n", "pass", "score", "trunc", "trunc&pass", "p50 tok", "p50 chars"))
print("-" * 70)
for t, _ in ARMS:
    d = arms[t]
    tk = sorted(v[0] for v in d.values() if isinstance(v[0], int))
    ch = sorted(v[3] for v in d.values())
    np_ = sum(1 for v in d.values() if v[2]); nt = sum(1 for v in d.values() if v[1])
    print("%-7s %4d %5d %8.4f %6d %11d %9d %10d"
          % (t, len(d), np_, np_ / len(d), nt, sum(1 for v in d.values() if v[1] and v[2]),
             tk[len(tk) // 2] if tk else 0, ch[len(ch) // 2]))

print("\n=== paired: problems UNTRUNCATED IN BOTH ARMS (the capability question) ===")
for a, b in (("armJ", "gepo1"), ("armJ", "gepo2"), ("gepo1", "gepo2")):
    da, db = arms[a], arms[b]
    common = sorted(set(da) & set(db))
    both = [t for t in common if not da[t][1] and not db[t][1]]
    n = len(both) or 1
    pa = sum(1 for t in both if da[t][2]); pb = sum(1 for t in both if db[t][2])
    oa = sum(1 for t in both if da[t][2] and not db[t][2])
    ob = sum(1 for t in both if db[t][2] and not da[t][2])
    ra = sum(1 for t in common if da[t][2]) / len(common)
    rb = sum(1 for t in common if db[t][2]) / len(common)
    print("%5s vs %-6s n_both_untrunc=%2d  %s=%.4f  %s=%.4f  delta=%+.2fpp  disc %d/%d  p=%.4f"
          % (a, b, len(both), a, pa / n, b, pb / n, (pb - pa) / n * 100, oa, ob, mcnemar(oa, ob)))
    print("%12s raw all n=%d: %.4f vs %.4f = %+.2fpp   <- how much is the cap?"
          % ("", len(common), ra, rb, (rb - ra) * 100))
    # paired LENGTH on the both-untruncated set (truncated rows sit AT the cap by
    # construction, so including them drags both arms to the ceiling and masks a real
    # difference -- the cap-asymmetry trap).
    #
    # TOKENS IS PRIMARY. Measured 2026-08-26 on the full armJ-vs-gepo1 cells, the two
    # channels DISAGREE IN DIRECTION: chars +507 median (gepo1 longer, p=0.55) but
    # tokens -120 median (gepo1 shorter, p=0.072), because chars/token differs per arm
    # (armJ 2.95, gepo1 3.04). A denser token is not a longer output. Length for a
    # brevity intervention means TOKENS -- what is generated, what the cap is
    # denominated in, what a length objective targets. Chars are reported alongside
    # ONLY so a divergence stays visible instead of being silently picked.
    for lbl, idx in (("tokens", 0), ("chars", 3)):
        va = [da[t][idx] for t in both]; vb = [db[t][idx] for t in both]
        if any(x is None for x in va + vb):
            print("%12s LENGTH %s: field missing on some rows -- SKIPPED" % ("", lbl)); continue
        dl = sorted(vb[i] - va[i] for i in range(len(both)))
        sh = sum(1 for x in dl if x < 0); ties = sum(1 for x in dl if x == 0)
        ma = sorted(va); mb = sorted(vb)
        pa50 = ma[len(ma) // 2]; pb50 = mb[len(mb) // 2]
        print("%12s LENGTH %-6s %s p50=%-7d %s p50=%-7d (%+.1f%%) median delta=%+d  "
              "%s shorter on %d/%d  sign p=%.4f%s"
              % ("", lbl, a, pa50, b, pb50, (pb50 - pa50) / pa50 * 100,
                 dl[len(dl) // 2], b, sh, len(dl) - ties,
                 sign_test(sh, len(dl) - ties), "   <- PRIMARY" if lbl == "tokens" else ""))
    ra = sum(da[t][3] for t in both) / max(1, sum(da[t][0] for t in both))
    rb = sum(db[t][3] for t in both) / max(1, sum(db[t][0] for t in both))
    print("%12s chars/token: %s=%.2f  %s=%.2f%s"
          % ("", a, ra, b, rb, "   (differ -> channels can disagree)" if abs(ra - rb) > 0.02 else ""))

print("\n=== where the truncations sit (finish_reason == 'length') ===")
tr = {t: {k for k, v in arms[t].items() if v[1]} for t, _ in ARMS}
for t, _ in ARMS:
    print("  %-6s truncated %d/%d" % (t, len(tr[t]), len(arms[t])))
print("  armJ-only %d (gepo2 PASSES %d) | gepo2-only %d (armJ PASSES %d) | both %d"
      % (len(tr["armJ"] - tr["gepo2"]), sum(1 for t in tr["armJ"] - tr["gepo2"] if arms["gepo2"][t][2]),
         len(tr["gepo2"] - tr["armJ"]), sum(1 for t in tr["gepo2"] - tr["armJ"] if arms["armJ"][t][2]),
         len(tr["armJ"] & tr["gepo2"])))
