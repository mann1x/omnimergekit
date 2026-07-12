#!/usr/bin/env python3
"""gpqa_gap_subset.py — define G_gap = GPQA-Diamond questions that v7-coder
gets RIGHT but v8 (fkbroad-soft2) gets WRONG (the ~20pp regression set), from
two lm-eval samples files. Matches by doc_id, scores on the flexible-extract
filter (the canonical GPQA filter). Emits the gap doc_ids + question metadata
so each recipe-dissection rung can be scored on exactly this subset.

Usage:
  gpqa_gap_subset.py <v7_samples.jsonl> <v8_samples.jsonl> <out.json>
"""
import json
import sys
from collections import Counter

v7_path, v8_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
FILTER = "flexible-extract"


def load_correct(path):
    """doc_id -> (exact_match==1.0) on the flexible-extract filter; plus docs."""
    correct, docs = {}, {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("filter") != FILTER:
                continue
            did = r["doc_id"]
            correct[did] = (r.get("exact_match") == 1.0)
            docs[did] = r.get("doc", {})
    return correct, docs


v7, v7docs = load_correct(v7_path)
v8, v8docs = load_correct(v8_path)
ids = sorted(set(v7) & set(v8))
print(f"v7 flex records: {len(v7)}  v8 flex records: {len(v8)}  matched doc_ids: {len(ids)}")
print(f"v7 score: {sum(v7.values())}/{len(v7)} = {sum(v7.values())/len(v7):.4f}")
print(f"v8 score: {sum(v8.values())}/{len(v8)} = {sum(v8.values())/len(v8):.4f}")

# 2x2 contingency
both_right = [d for d in ids if v7[d] and v8[d]]
v7_only = [d for d in ids if v7[d] and not v8[d]]   # == G_gap
v8_only = [d for d in ids if not v7[d] and v8[d]]   # reverse gap
both_wrong = [d for d in ids if not v7[d] and not v8[d]]
print("\n== contingency (flexible-extract) ==")
print(f"  both_right : {len(both_right)}")
print(f"  v7_only(G_gap): {len(v7_only)}   <-- v7 right, v8 wrong (the regression)")
print(f"  v8_only    : {len(v8_only)}   (v8 right, v7 wrong)")
print(f"  both_wrong : {len(both_wrong)}")
print(f"  net v7-v8 : {len(v7_only)-len(v8_only)} questions ({(len(v7_only)-len(v8_only))/len(ids)*100:.2f}pp)")

# domain breakdown of G_gap
def dom(d):
    doc = v8docs.get(d) or v7docs.get(d) or {}
    return doc.get("High-level domain") or doc.get("Subdomain") or "?"

gap_dom = Counter(dom(d) for d in v7_only)
print("\n== G_gap domain breakdown ==")
for k, n in gap_dom.most_common():
    print(f"  {k:<28} {n}")

# emit G_gap
out = {
    "filter": FILTER,
    "v7_samples": v7_path,
    "v8_samples": v8_path,
    "v7_score": sum(v7.values()) / len(v7),
    "v8_score": sum(v8.values()) / len(v8),
    "gap_doc_ids": v7_only,
    "reverse_gap_doc_ids": v8_only,
    "gap": [
        {
            "doc_id": d,
            "domain": dom(d),
            "answer": (v8docs.get(d) or {}).get("answer"),
            "question": ((v8docs.get(d) or {}).get("Question") or "")[:200],
        }
        for d in v7_only
    ],
}
json.dump(out, open(out_path, "w"), indent=1)
print(f"\n== wrote {out_path}: G_gap = {len(v7_only)} doc_ids ==")
print("gap_doc_ids:", v7_only)
