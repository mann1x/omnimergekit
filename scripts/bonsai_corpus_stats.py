#!/usr/bin/env python3
"""Completeness and length table of a generated replay set:  bonsai_corpus_stats.py PLAN.jsonl GEN_PREFIX

Rows = family / mode / effort. kept = complete conversations; drops by reason; tokens of the kept ones."""
import collections
import json
import re
import sys

plan = {r["id"]: r for r in map(json.loads, open(sys.argv[1]))}
kept = [json.loads(x) for x in open(sys.argv[2] + ".jsonl")]
drop = [json.loads(x) for x in open(sys.argv[2] + ".dropped.jsonl")]
key = lambda i: (plan[i]["fam"], plan[i]["mode"], plan[i].get("effort", ""))   # noqa: E731
rows = collections.defaultdict(lambda: dict(kept=0, tok=[], why=collections.Counter(), think=[], cjk=0, tool=0, fence_odd=0))
for r in kept:
    q = rows[key(r["id"])]
    gen = r["text"].split("<|im_start|>assistant\n")[-1]
    q["kept"] += 1
    q["tok"].append(r["gen_tokens"])
    q["think"].append(r["think_chars"])
    q["cjk"] += bool(re.search(r"[一-鿿]", gen))
    q["tool"] += r["tool_call"]
    q["fence_odd"] += gen.count("```") % 2
for r in drop:
    rows[key(r["id"])]["why"][r["why"]] += 1
print(f"{'family':8s} {'mode':8s} {'effort':7s} {'kept':>9s}  {'tokens p50/p90/max':>20s} {'think ch p50':>12s}  cjk odd-fence tool  drops")
tk = td = tt = 0
for k in sorted(rows):
    q = rows[k]
    n = q["kept"] + sum(q["why"].values())
    t = sorted(q["tok"]) or [0]
    th = sorted(q["think"]) or [0]
    tk, td, tt = tk + q["kept"], td + n - q["kept"], tt + sum(q["tok"])
    print(f"{k[0]:8s} {k[1]:8s} {k[2] or '-':7s} {q['kept']:4d}/{n:<4d}  {t[len(t) // 2]:6d}/{t[int(len(t) * .9)]:6d}/{t[-1]:6d} {th[len(th) // 2]:12d}  "
          f"{q['cjk']:3d} {q['fence_odd']:9d} {q['tool']:4d}  {dict(q['why']) or ''}")
print(f"total kept {tk}, dropped {td} ({100 * tk / max(1, tk + td):.1f}% kept), {tt} generated tokens kept, mean {tt / max(1, tk):.0f}")
