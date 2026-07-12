#!/usr/bin/env python3
"""Audit GPQA flexible-extract scoring: is a 'wrong' record a genuine
wrong-letter pick, or an answer the flexible-extract filter failed to pull?

Usage: gpqa_extract_audit.py <samples.jsonl> [dump_n]

Resolves the 198-questions x 2-filters (strict/flexible) record layout,
recomputes the flexible-extract score, categorizes every wrong record,
and runs a permissive re-extraction to bound how many answers the
shipped regex missed. Dumps raw tails for the no-extraction cases so a
human can eyeball whether the model actually answered.
"""
import json
import re
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
dump_n = int(sys.argv[2]) if len(sys.argv) > 2 else 8

recs = []
with open(path) as fh:
    for line in fh:
        line = line.strip()
        if line:
            recs.append(json.loads(line))

print(f"FILE={path}")
print(f"total_lines={len(recs)}")
print(f"record_keys={sorted(recs[0].keys())}")

# ---- group by filter -------------------------------------------------------
# lm_eval writes one record per (doc, filter). Detect the filter key.
filt_key = "filter" if "filter" in recs[0] else None
by_filter = defaultdict(list)
for r in recs:
    fk = r.get(filt_key) if filt_key else "?"
    by_filter[fk].append(r)
print(f"filter_key={filt_key}  filters={ {k: len(v) for k,v in by_filter.items()} }")

def get_score(r):
    # exact_match may be top-level or under a metrics dict
    if "exact_match" in r:
        return r["exact_match"]
    for k, v in r.items():
        if k.startswith("exact_match"):
            return v
    return None

def target_letter(r):
    t = r.get("target")
    if isinstance(t, str):
        m = re.search(r"[A-D]", t)
        if m:
            return m.group(0)
        return t
    return str(t)

def extracted(r):
    fr = r.get("filtered_resps")
    if isinstance(fr, list) and fr:
        return str(fr[0])
    return str(fr)

def raw_text(r):
    rp = r.get("resps")
    # resps is typically [[text]]
    while isinstance(rp, list) and rp:
        rp = rp[0]
    return rp if isinstance(rp, str) else str(rp)

# Permissive re-extraction: try several answer-marker patterns, take last hit.
PATTERNS = [
    re.compile(r"answer is[:\s]*\(?([A-D])\)?", re.I),
    re.compile(r"answer[:\s]*\(?([A-D])\)?\b", re.I),
    re.compile(r"\*\*\(?([A-D])\)?\*\*"),
    re.compile(r"\\boxed\{\(?([A-D])\)?\}"),
    re.compile(r"\b([A-D])\b(?=[^A-Za-z]*$)"),  # trailing lone letter
    re.compile(r"\(([A-D])\)"),                  # any (X), last one
]
def permissive(txt):
    best = None
    for pat in PATTERNS:
        for m in pat.finditer(txt):
            best = m.group(1).upper()
        if best:
            return best
    return None

flex = by_filter.get("flexible-extract") or by_filter.get("flexible_extract")
if flex is None:
    # fall back: maybe only one filter present
    flex = recs
    print("WARN: no flexible-extract filter key; treating all records as one set")

n = len(flex)
correct = sum(1 for r in flex if get_score(r) == 1.0)
print(f"\n=== flexible-extract: n={n} correct={correct} score={correct/n:.4f} ===")

# strict for reference
strict = by_filter.get("strict-match") or by_filter.get("strict_match")
if strict:
    sc = sum(1 for r in strict if get_score(r) == 1.0)
    print(f"=== strict-match:    n={len(strict)} correct={sc} score={sc/len(strict):.4f} ===")

# categorize wrong flexible records
wrong = [r for r in flex if get_score(r) != 1.0]
valid_letters = set("ABCD")
genuine_wrong = []   # extracted a valid A-D, but != target
no_extract = []      # filter pulled nothing usable
extracted_dist = Counter()
for r in wrong:
    ex = extracted(r).strip()
    ex_letter = None
    m = re.search(r"[A-D]", ex)
    if m and len(ex) <= 4:        # a clean letter-ish extraction
        ex_letter = m.group(0)
    extracted_dist[ex if len(ex) <= 6 else f"<len{len(ex)}>"] += 1
    if ex_letter in valid_letters:
        genuine_wrong.append((r, ex_letter))
    else:
        no_extract.append(r)

print(f"\nwrong={len(wrong)}  genuine-wrong(valid letter != target)={len(genuine_wrong)}"
      f"  no-clean-extraction={len(no_extract)}")
print(f"extracted-value distribution on wrong records: {dict(extracted_dist.most_common(12))}")

# permissive recovery: of the wrong records, how many would a permissive
# re-extraction now score CORRECT (its recovered letter == target)?
recoverable = 0
recover_examples = []
for r in wrong:
    tgt = target_letter(r)
    rec = permissive(raw_text(r))
    if rec is not None and rec == tgt:
        recoverable += 1
        if len(recover_examples) < dump_n:
            recover_examples.append((r, rec, tgt))
print(f"\nPERMISSIVE recovery on wrong records: {recoverable} "
      f"(upper-bound score if all recovered = {(correct+recoverable)/n:.4f})")

print(f"\n----- up to {dump_n} NO-CLEAN-EXTRACTION raw tails (target letter shown) -----")
for r in no_extract[:dump_n]:
    tgt = target_letter(r)
    txt = raw_text(r)
    rec = permissive(txt)
    tail = txt[-360:].replace("\n", " ")
    print(f"\n[doc {r.get('doc_id')}] target={tgt}  flex_extracted={extracted(r)!r}  permissive={rec}")
    print(f"  tail: ...{tail}")

print(f"\n----- up to {dump_n} GENUINE-WRONG samples (extracted valid letter != target) -----")
for r, exl in genuine_wrong[:dump_n]:
    tgt = target_letter(r)
    txt = raw_text(r)
    tail = txt[-260:].replace("\n", " ")
    print(f"\n[doc {r.get('doc_id')}] target={tgt}  extracted={exl}")
    print(f"  tail: ...{tail}")
