"""Paired per-question comparison for ONE lm-eval-backed qwen_suite bench.

Usage: paired_cell.py <bench>

WHY: a headline pp delta at n=100..198 cannot separate a real regression from a draw. The
discordant counts are the content. Also reports completion-token distribution, because GEPO
is a brevity intervention and "did the outputs actually get shorter" is a separate question
from "did the score move".

The metric+filter are read from each cell's OWN summary.json (omk already resolved
flexible-extract / math_verify / etc). They are NOT hardcoded, and a disagreement between
arms is fatal -- comparing two arms scored on different keys is not a comparison.

Refuses loudly on non-lm-eval benches (lcb/multipl_e/humaneval keep their own samples
layout); use lcb_captab.py / mpe_captab.py for those.
"""
import glob, json, sys
from math import comb
from transformers import AutoTokenizer

BENCH = sys.argv[1]
R = "/srv/ml/eval_results/qwen_suite/" + BENCH
TOK = "/mnt/sdc/ml/brevity/gepo/a3b-gepo2"
ARMS = [("armJ", "qwenhybridp24_q6k"),
        ("gepo1", "qwena3bgepo1_q6k"),
        ("gepo2", "qwena3bgepo2_q6k")]

tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(name):
    s = json.load(open(R + "/" + name + "/summary.json"))
    metric, filt = s.get("metric"), s.get("filter")
    f = sorted(glob.glob(R + "/" + name + "/lm_eval_out/**/samples_*.jsonl", recursive=True))
    if not f:
        raise SystemExit("REFUSE: no lm-eval samples for %s/%s -- not an lm-eval bench? "
                         "use lcb_captab.py / mpe_captab.py" % (BENCH, name))
    out = {}
    for line in open(f[-1]):
        if not line.strip():
            continue
        d = json.loads(line)
        if filt is not None and d.get("filter") != filt:
            continue
        if metric not in d:
            raise SystemExit("REFUSE: metric %r absent from samples of %s/%s (keys=%s)"
                             % (metric, BENCH, name, sorted(d.keys())))
        r = (d.get("resps") or [[""]])[0]
        txt = r[0] if r else ""
        out[d["doc_id"]] = (bool(d[metric]),
                            len(tok(txt, add_special_tokens=False)["input_ids"]))
    return out, s.get("score"), metric, filt


data, meta = {}, {}
for t, n in ARMS:
    data[t], sc, m, fl = load(n)
    meta[t] = (sc, m, fl)

keys = {(m, fl) for _, m, fl in [(a,) + meta[t][1:] for a, t in zip(range(3), data)]}
mset = {meta[t][1:] for t in data}
if len(mset) != 1:
    raise SystemExit("REFUSE: arms scored on different metric/filter: %s" % mset)
print("bench=%s  metric=%s  filter=%s" % (BENCH, meta["armJ"][1], meta["armJ"][2]))
print()
print("%-7s %5s %6s %9s %9s %9s %9s" % ("arm", "n", "pass", "score", "p50 tok", "mean tok", "sum tok"))
print("-" * 62)
for t, _ in ARMS:
    d = data[t]
    toks = sorted(v[1] for v in d.values())
    npass = sum(1 for v in d.values() if v[0])
    print("%-7s %5d %6d %9.4f %9d %9.0f %9d"
          % (t, len(d), npass, npass / len(d), toks[len(toks) // 2],
             sum(toks) / len(toks), sum(toks)))
    if abs(npass / len(d) - meta[t][0]) > 1e-6:
        print("        !!! recomputed %.4f != summary.json %.4f" % (npass / len(d), meta[t][0]))

print()
print("=== paired McNemar (exact two-sided) ===")
for a, b in (("armJ", "gepo1"), ("armJ", "gepo2"), ("gepo1", "gepo2")):
    da, db = data[a], data[b]
    common = sorted(set(da) & set(db))
    pa = sum(1 for q in common if da[q][0])
    pb = sum(1 for q in common if db[q][0])
    oa = sum(1 for q in common if da[q][0] and not db[q][0])
    ob = sum(1 for q in common if db[q][0] and not da[q][0])
    n = len(common)
    print("%5s vs %-6s n=%3d  %s=%3d (%.4f)  %s=%3d (%.4f)  delta=%+.2fpp  "
          "discordant %d/%d  p=%.4f"
          % (a, b, n, a, pa, pa / n, b, pb, pb / n, (pb - pa) / n * 100, oa, ob,
             mcnemar(oa, ob)))
