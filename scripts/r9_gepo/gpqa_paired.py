"""Paired GPQA comparison across armJ / gepo1 / gepo2 + output-length stats.

WHY: gepo2 scores 0.7576 vs gepo1 0.7980 vs armJ 0.8333. GEPO run 2 is a BREVITY
intervention, so the hypothesis with teeth is that shorter reasoning costs accuracy on a
reasoning bench. A raw pp delta cannot separate "worse model" from "noise at n=198", and it
says nothing about mechanism. So: McNemar on the paired per-question outcomes (discordant
counts are the real content), plus completion-token distribution per arm.

Scoring rows: lm-eval writes one row per (doc, filter). Only flexible-extract is canonical
for GPQA (per omk doctrine); strict-match rows are dropped.
"""
import glob, json
from math import comb
from transformers import AutoTokenizer

R = "/srv/ml/eval_results/qwen_suite/gpqa_diamond_full"
TOK = "/mnt/sdc/ml/brevity/gepo/a3b-gepo2"
ARMS = [("armJ", "qwenhybridp24_q6k"),
        ("gepo1", "qwena3bgepo1_q6k"),
        ("gepo2", "qwena3bgepo2_q6k")]
FILT = "flexible-extract"

tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(name):
    f = sorted(glob.glob(R + "/" + name + "/lm_eval_out/**/samples_*.jsonl", recursive=True))[-1]
    out = {}
    for line in open(f):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("filter") != FILT:
            continue
        r = (d.get("resps") or [[""]])[0]
        txt = r[0] if r else ""
        out[d["doc_id"]] = (bool(d.get("exact_match")),
                            len(tok(txt, add_special_tokens=False)["input_ids"]))
    return out


arms = {t: load(n) for t, n in ARMS}

print("%-7s %5s %6s %9s %9s %9s %9s" % ("arm", "n", "pass", "score", "p50 tok", "mean tok", "sum tok"))
print("-" * 62)
for t, _ in ARMS:
    d = arms[t]
    toks = sorted(v[1] for v in d.values())
    npass = sum(1 for v in d.values() if v[0])
    print("%-7s %5d %6d %9.4f %9d %9.0f %9d"
          % (t, len(d), npass, npass / len(d), toks[len(toks) // 2],
             sum(toks) / len(toks), sum(toks)))

print()
print("=== paired McNemar (same 198 questions, exact two-sided) ===")
for a, b in (("armJ", "gepo1"), ("armJ", "gepo2"), ("gepo1", "gepo2")):
    da, db = arms[a], arms[b]
    common = sorted(set(da) & set(db))
    pa = sum(1 for q in common if da[q][0])
    pb = sum(1 for q in common if db[q][0])
    only_a = sum(1 for q in common if da[q][0] and not db[q][0])
    only_b = sum(1 for q in common if db[q][0] and not da[q][0])
    n = len(common)
    print("%5s vs %-6s n=%d  %s=%d (%.4f)  %s=%d (%.4f)  delta=%+.2fpp  "
          "discordant %s-only=%d %s-only=%d  p=%.4f"
          % (a, b, n, a, pa, pa / n, b, pb, pb / n, (pb - pa) / n * 100,
             a, only_a, b, only_b, mcnemar(only_a, only_b)))

print()
print("=== length vs correctness (is the drop a BREVITY effect?) ===")
for t, _ in ARMS:
    d = arms[t]
    ok = [v[1] for v in d.values() if v[0]]
    no = [v[1] for v in d.values() if not v[0]]
    print("%-7s correct: n=%3d mean_tok=%6.0f   wrong: n=%3d mean_tok=%6.0f"
          % (t, len(ok), sum(ok) / len(ok), len(no), sum(no) / len(no)))
