#!/usr/bin/env python3
"""Paired per-question GPQA: published v6 Q6_K vs AC-imatrix Q6_K, BOTH GREEDY.

Greedy removes the sampler noise that made the earlier sampled comparison unreadable, so
per-question disagreement here is attributable to the quant, not to the draw. McNemar's
exact test asks whether the disagreement is ASYMMETRIC.

Also counts degenerate outputs ('[invalid]' extraction, empty generations) -- those would
mean a broken quant rather than merely a worse-scoring one, and the mean cannot see them.
"""
import json
import sys
from math import comb

B = "/srv/ml/eval_results/v6_greedy_gpqa"
PUB = (B + "/v6pub_q6k_greedy/gpqa_diamond_full/v6pub_q6k_greedy/lm_eval_out/"
       "v6pub_q6k_greedy/samples_gpqa_diamond_cot_zeroshot_"
       "2026-09-09T06-22-18.808799.jsonl")
AC = (B + "/v6ac_q6k_greedy/gpqa_diamond_full/v6ac_q6k_greedy/lm_eval_out/"
      "v6ac_q6k_greedy/samples_gpqa_diamond_cot_zeroshot_"
      "2026-09-09T06-24-56.600145.jsonl")
FILT = "flexible-extract"


def load(path):
    """One row per (doc_id, filter) -- 396 rows for 198 questions.

    Selecting on r["filter"] is MANDATORY: keying on doc_id alone silently keeps whichever
    filter was written last, and strict-match scores 0.0 on this bench, so the wrong pick
    reports 0%. The degenerate count is only meaningful within the scored filter for the
    same reason -- every strict-match row is '[invalid]' by construction.
    """
    out, degen, lens, seen = {}, 0, [], 0
    for line in open(path):
        r = json.loads(line)
        if r.get("filter") != FILT:
            continue
        seen += 1
        key = r["doc_id"]
        if key in out:
            print("*** duplicate doc_id %s within filter %s" % (key, FILT))
        resp = r.get("resps") or [[""]]
        txt = resp[0][0] if resp and resp[0] else ""
        lens.append(len(txt))
        if not txt.strip() or "[invalid]" in str(r.get("filtered_resps", "")):
            degen += 1
        out[key] = int(round(float(r["exact_match"])))
    print("  %s: %d rows at filter=%s" % (path.rsplit("/", 1)[-1][:40], seen, FILT))
    return out, degen, lens


def mcnemar_exact(b, c):
    """Two-sided exact binomial on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def main():
    pub, dpub, lpub = load(PUB)
    ac, dac, lac = load(AC)
    keys = sorted(set(pub) & set(ac))
    print("paired questions: %d  (pub=%d ac=%d)" % (len(keys), len(pub), len(ac)))
    if len(keys) != len(pub) or len(keys) != len(ac):
        print("*** doc_id sets differ -- NOT a clean pairing")

    a = sum(1 for k in keys if pub[k] and ac[k])
    b = sum(1 for k in keys if pub[k] and not ac[k])
    c = sum(1 for k in keys if not pub[k] and ac[k])
    d = sum(1 for k in keys if not pub[k] and not ac[k])
    n = len(keys)
    print("\n              ac_correct  ac_wrong")
    print("pub_correct   %6d      %6d" % (a, b))
    print("pub_wrong     %6d      %6d" % (c, d))
    print("\npub  = %d/%d = %.2f%%" % (a + b, n, 100.0 * (a + b) / n))
    print("ac   = %d/%d = %.2f%%" % (a + c, n, 100.0 * (a + c) / n))
    print("delta(ac-pub) = %+.2f pp" % (100.0 * (c - b) / n))
    print("discordant    = %d/%d = %.1f%%  <- the noise floor for this pair"
          % (b + c, n, 100.0 * (b + c) / n))
    print("McNemar exact p = %.4f  (b=%d pub-only-right, c=%d ac-only-right)"
          % (mcnemar_exact(b, c), b, c))

    # 95% CI on the PAIRED difference (Agresti-Min). A plain two-proportion CI would be
    # wrong here -- the arms answer the same 198 questions, so the errors are correlated.
    var = (b + c - float(b - c) ** 2 / n) / float(n) ** 2
    se = var ** 0.5
    lo, hi = ((c - b) / float(n) - 1.95996 * se), ((c - b) / float(n) + 1.95996 * se)
    print("95%% CI on delta = [%+.2f, %+.2f] pp   (SE=%.2f pp)"
          % (100 * lo, 100 * hi, 100 * se))
    print("  -> a 5 pp DEGRADATION (delta = -5.00) is %s this interval"
          % ("INSIDE" if lo <= -0.05 <= hi else "OUTSIDE -- ruled out at 95%"))

    print("\ndegenerate outputs: pub=%d  ac=%d  (empty or [invalid])" % (dpub, dac))
    for nm, ls in (("pub", lpub), ("ac", lac)):
        ls = sorted(ls)
        print("  %s resp chars: p50=%d p90=%d max=%d"
              % (nm, ls[len(ls) // 2], ls[int(len(ls) * 0.9)], ls[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
