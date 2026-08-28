"""LCB-48k truncation cross-tab + paired both-untruncated comparison.

WHY: gepo2 scores 64/77 vs armJ 56/77 = +8 problems, and gepo2 truncates at the
49152 wall 7 times vs armJ's 15 = 8 fewer. A completion cut at the ceiling emits
unparsable code and fails, so an exact match between "problems gained" and
"truncations avoided" is the signature of a score delta that is really a LENGTH
delta. That is exactly what happened to gepo1 on MPE-100 (8 fewer caps, won by
exactly 8, +0.00pp on the both-uncapped set).

The coincidence does NOT by itself prove it — the two counts can match while the
underlying problems differ. Only the paired both-untruncated set answers the
capability question. Build it.

CORRECTION (verified 2026-08-26): an earlier version of this docstring claimed
"unlike MPE, LCB's summary DOES report finish_reasons". That is FALSE. Measured
directly, summary.json finish_reasons is None on LCB in BOTH cohorts -- ream_arms
lcb_v6_77q_48k AND qwen_suite lcb_v6_77q. omk's truncation signal is blind on LCB
universally, exactly as it is on MPE (bug-592). The numbers this script produces
were never affected, because it re-derives truncation from token counts and never
consulted finish_reasons -- but do not repeat the claim.

Truncation is re-derived per problem the same way omk defines it: tokens of the
completion, add_special_tokens=False, >= max_gen_toks - 16.
"""
import json
import sys
from math import comb

from transformers import AutoTokenizer

R = "/srv/ml/eval_results/ream_arms/lcb_v6_77q_48k"
TOKENIZER = "/mnt/sdc/ream-work/armJ"
MAX_GEN_TOKS = 49152
TOL = 16
ARMS = [("armJ", "lcb48k_armJ_hybrid_p24"),
        ("gepo1", "lcb48k_a3b_gepo1"),
        ("gepo2", "lcb48k_a3b_gepo2")]

tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def load(name):
    """-> {task_id: (n_tokens, truncated, passed)}"""
    out = {}
    with open(f"{R}/{name}/lcb_result.samples.jsonl") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            tid = d.get("task_id") or d.get("doc_id")
            text = d.get("completion") or ""
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            out[tid] = (n, n >= MAX_GEN_TOKS - TOL, bool(d.get("passed")))
    return out


arms = {tag: load(name) for tag, name in ARMS}

print(f"{'arm':<7} {'n':>4} {'pass':>5} {'score':>8} {'trunc':>6} "
      f"{'trunc&pass':>11} {'p50 tok':>8} {'mean tok':>9} {'sum tok':>10}")
print("-" * 76)
for tag, _ in ARMS:
    d = arms[tag]
    toks = sorted(v[0] for v in d.values())
    npass = sum(1 for v in d.values() if v[2])
    ntr = sum(1 for v in d.values() if v[1])
    trpass = sum(1 for v in d.values() if v[1] and v[2])
    print(f"{tag:<7} {len(d):>4} {npass:>5} {npass/len(d):>8.4f} {ntr:>6} "
          f"{trpass:>11} {toks[len(toks)//2]:>8} {sum(toks)/len(toks):>9.0f} "
          f"{sum(toks):>10}")

print()
print("=== paired: problems UNTRUNCATED IN BOTH ARMS (the capability question) ===")
for a, b in (("armJ", "gepo1"), ("armJ", "gepo2"), ("gepo1", "gepo2")):
    da, db = arms[a], arms[b]
    common = sorted(set(da) & set(db))
    both = [t for t in common if not da[t][1] and not db[t][1]]
    pa = sum(1 for t in both if da[t][2])
    pb = sum(1 for t in both if db[t][2])
    d_a = sum(1 for t in both if da[t][2] and not db[t][2])
    d_b = sum(1 for t in both if db[t][2] and not da[t][2])
    n = len(both)
    ra = sum(1 for t in common if da[t][2]) / len(common)
    rb = sum(1 for t in common if db[t][2]) / len(common)
    print(f"{a:>5} vs {b:<6} n_both_untrunc={n:>3}  "
          f"{a}={pa}/{n} ({pa/n:.4f})  {b}={pb}/{n} ({pb/n:.4f})  "
          f"delta={((pb-pa)/n)*100:+.2f}pp  discordant {d_a}/{d_b}  "
          f"p={mcnemar(d_a, d_b):.4f}")
    print(f"{'':>5}    {'':<6} raw all n={len(common)}: {ra:.4f} vs {rb:.4f} "
          f"= {(rb-ra)*100:+.2f}pp")

print()
print("=== where the truncations sit ===")
tr = {tag: {t for t, v in arms[tag].items() if v[1]} for tag, _ in ARMS}
print(f"  armJ trunc set  : {len(tr['armJ'])}")
print(f"  gepo2 trunc set : {len(tr['gepo2'])}")
print(f"  armJ-only trunc : {len(tr['armJ'] - tr['gepo2'])}  "
      f"(of which gepo2 PASSES: "
      f"{sum(1 for t in tr['armJ'] - tr['gepo2'] if arms['gepo2'][t][2])})")
print(f"  gepo2-only trunc: {len(tr['gepo2'] - tr['armJ'])}  "
      f"(of which armJ PASSES: "
      f"{sum(1 for t in tr['gepo2'] - tr['armJ'] if arms['armJ'][t][2])})")
print(f"  both trunc      : {len(tr['armJ'] & tr['gepo2'])}")
