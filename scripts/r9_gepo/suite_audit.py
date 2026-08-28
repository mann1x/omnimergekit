"""Cap / truncation / empty audit across the 10 qwen_suite bench pairs.

The 2026-08-24 MPE lesson: a ceiling that binds asymmetrically turns a bench into a
length meter. Check every pair before reading any delta as capability.
"""
import json, os

R = "/srv/ml/eval_results/qwen_suite"
NEW, REF = "qwena3bgepo1_q6k", "qwenhybridp24_q6k"
OVERRIDE = {"multipl_e_100": REF + "_b604"}
BENCH = ["gpqa_diamond_full", "gsm8k_100_boxed", "arc_challenge_100",
         "humaneval_full_think", "humanevalplus_full_think", "multipl_e_100",
         "ifeval_100", "math500_100_qwen", "aime_30_qwen", "lcb_v6_77q"]


def load(b, cell):
    try:
        return json.load(open(os.path.join(R, b, cell, "summary.json")))
    except Exception:
        return None


hdr = ("%-26s %-6s %8s %6s %8s %8s %7s %7s %8s %8s"
       % ("bench", "arm", "score", "n", "capped", "cap%", "trunc", "empty", "p50", "p90"))
print(hdr)
print("-" * len(hdr))
flags = []
for b in BENCH:
    row = {}
    for tag, cell in (("armJ", OVERRIDE.get(b, REF)), ("gepo1", NEW)):
        d = load(b, cell)
        if not d:
            print("%-26s %-6s %8s" % (b, tag, "MISSING")); continue
        g = d.get("generation_caps") or {}
        t = d.get("token_stats") or {}
        c = t.get("completion_tokens") or {}
        fr = t.get("finish_reasons") or {}
        cap = g.get("capped_total")
        row[tag] = (cap, g.get("verdict"), g.get("ceiling_hit"), g.get("max_gen_toks"))
        print("%-26s %-6s %8.4f %6s %8s %7s %7s %7s %8s %8s"
              % (b, tag, d.get("score"), t.get("n"),
                 cap if cap is not None else "-",
                 ("%.1f%%" % g["capped_pct"]) if g.get("capped_pct") is not None else "-",
                 fr.get("length", 0), t.get("empty_completions", 0),
                 c.get("p50"), c.get("p90")))
    if "armJ" in row and "gepo1" in row:
        ca, cb = row["armJ"][0], row["gepo1"][0]
        if ca is not None and cb is not None and (ca or cb):
            if abs(ca - cb) >= 3:
                flags.append("%s: cap asymmetry armJ=%d gepo1=%d (ceiling %s, max_gen_toks=%s)"
                             % (b, ca, cb, row["armJ"][2], row["armJ"][3]))
            elif ca or cb:
                flags.append("%s: both capped but symmetric armJ=%d gepo1=%d" % (b, ca, cb))
    print()

print("=" * 70)
if flags:
    print("CAP FLAGS:")
    for f in flags:
        print("  -", f)
else:
    print("No cell on either arm hit a declared ceiling: every delta is cap-free.")
