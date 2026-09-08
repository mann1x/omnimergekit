#!/usr/bin/env python
"""One mechanism for two anomalies? Brace alignment against the prompt's trailing '{'.

Every MultiPL-E prompt ends with an OPEN function body ('... {\\n'). The completion must
supply the body and NOT the closing brace -- the bench appends it, and "\\n}" is a stop token.
Two separate anomalies from the audit look like the same failure once framed that way:

  armI "silent empty" (18/300)  -- completion is '\\n': the model closed the body immediately.
                                   Recorded as empty; actually an EMPTY BODY.
  armG java collapse (68/100)   -- completions begin INSIDE a block that was never opened,
                                   leaving unmatched '}'. An OVER-CLOSED body.

Both are the model mis-tracking how many blocks are open at the prompt boundary. If that is
one mechanism, the arms should separate on a brace-balance statistic computed over ALL 300
completions -- not just the failing ones -- and the direction should differ per arm.

Zero GPU: reads the saved completions only. Classifies each completion by
  close_first  : first non-blank line is just a closing brace  -> empty body
  neg_balance  : more closers than openers -> over-closed (armG's shape)
  balanced/pos : normal body
and reports, per arm and per language, next to that arm's measured pass rate.
"""
import glob
import json
import os
import re
import collections

R = "/srv/ml/eval_results/ream_arms/multipl_e_100"
LANGS = {"humaneval-rs": "rs", "humaneval-java": "java", "humaneval-js": "js"}
ARMS = [
    ("pub184e", "pub184e_imat"), ("armD", "reamD_ourssal_nomerge"),
    ("armF rnorm", "reamF_rnorm_nomerge"), ("armE", "reamE_reapsal_nomerge"),
    ("armI p12", "hybrid_p12_ourssal_reapfloor"),
    ("armJ p24", "hybrid_p24_ourssal_reapfloor"),
    ("armG gs2", "reamG_ourssal_merge_gs2"), ("armH gs4", "reamH_ourssal_merge_gs4"),
    ("armB", "reamB_reapsal_merge"), ("base256e", "base256e_imat"),
]
CLOSER = re.compile(r"^\s*[}\)\]];?\s*$")


def rows(cell):
    """task_id -> (completion, status)"""
    comp = {}
    p = os.path.join(R, cell, "mpe_result.samples.jsonl")
    if not os.path.exists(p):
        return {}
    for line in open(p):
        line = line.strip()
        if line:
            d = json.loads(line)
            comp[d["task_id"]] = d.get("completion") or ""
    st = {}
    for ld, lang in LANGS.items():
        for f in glob.glob(os.path.join(R, cell, "results", ld, "*.results.json")):
            if f.endswith("_summary.json"):
                continue
            d = json.load(open(f))
            r = d.get("results") or []
            if r:
                st["%s::%s" % (lang, d.get("name"))] = r[0].get("status", "?")
    return {k: (v, st.get(k, "?")) for k, v in comp.items()}


def cls(c):
    if not c.strip():
        return "EMPTY_BODY"          # '\n' + stop-truncated '\n}'
    first = next((l for l in c.split("\n") if l.strip()), "")
    if CLOSER.match(first):
        return "EMPTY_BODY"          # opened nothing, closed immediately
    bal = c.count("{") - c.count("}")
    if bal < 0:
        return "OVER_CLOSED"         # armG's shape: starts inside an unopened block
    return "BODY"


print("%-11s %-5s | %4s %10s %11s %6s | %s"
      % ("arm", "lang", "OK", "EMPTY_BODY", "OVER_CLOSED", "BODY", "note"))
print("-" * 84)
tot = {}
for label, cell in ARMS:
    d = rows(cell)
    if not d:
        print("%-11s (no samples on disk -- SKIPPED)" % label)
        continue
    agg = collections.Counter()
    for lang in ("rs", "java", "js"):
        ks = [k for k in d if k.startswith(lang + "::")]
        if not ks:
            continue
        c = collections.Counter(cls(d[k][0]) for k in ks)
        ok = sum(1 for k in ks if d[k][1] == "OK")
        agg.update(c)
        print("%-11s %-5s | %4d %10d %11d %6d |"
              % (label, lang, ok, c["EMPTY_BODY"], c["OVER_CLOSED"], c["BODY"]))
    tot[label] = agg
    print()

print("=" * 84)
print("%-11s %11s %12s   (n=300)" % ("arm", "EMPTY_BODY", "OVER_CLOSED"))
for label in tot:
    print("%-11s %11d %12d" % (label, tot[label]["EMPTY_BODY"], tot[label]["OVER_CLOSED"]))
print("\nIf ONE mechanism: arms separate on EMPTY_BODY+OVER_CLOSED combined, and the two")
print("columns trade off. If the columns are uncorrelated across arms, they are two bugs.")
