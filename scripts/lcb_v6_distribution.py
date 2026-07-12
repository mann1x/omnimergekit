#!/usr/bin/env python3
"""Analyze the LiveCodeBench release_v6 (code_generation_lite) distribution to
design a discriminating, curated LCB-v6-55 subset that is 'not too new, not too
hard'. Mirrors the omk loader's scorer-compatibility filter exactly:
functional testtype + class-based starter (def X(self...)) so score_lcb_problem
can run it. Prints difficulty x year-month histograms + the current saturated
pool size, so we can pick a date window and difficulty mix."""
import json, re, collections
from huggingface_hub import hf_hub_download

REL = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
SELF_RE = re.compile(r"def\s+(\w+)\s*\(\s*self")

rows = []
for fn in REL:
    try:
        p = hf_hub_download(repo_id="livecodebench/code_generation_lite",
                            repo_type="dataset", filename=fn)
    except Exception as e:
        print(f"skip {fn}: {e}"); continue
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
print(f"total rows across {len(REL)} release files: {len(rows)}")

# scorer-compatible filter: functional + class-based starter
def compat(r):
    pr = r.get("public_test_cases", "[]")
    if isinstance(pr, str):
        try: pub = json.loads(pr)
        except Exception: return False
    else:
        pub = pr or []
    if not pub or pub[0].get("testtype") != "functional":
        return False
    starter = r.get("starter_code", "") or ""
    return bool(SELF_RE.search(starter))

comp = [r for r in rows if compat(r)]
print(f"scorer-compatible (functional + class-based): {len(comp)}")

# difficulty counts
bydiff = collections.Counter(r.get("difficulty", "?") for r in comp)
print("by difficulty:", dict(bydiff))

# medium: histogram by year-month
def ym(r): return (r.get("contest_date") or "")[:7]
for diff in ("easy", "medium", "hard"):
    sub = [r for r in comp if r.get("difficulty") == diff]
    hist = collections.Counter(ym(r) for r in sub)
    print(f"\n=== {diff} (n={len(sub)}) by contest year-month ===")
    cum = 0
    for k in sorted(hist):
        cum += hist[k]
        print(f"  {k}: {hist[k]:3d}   (cum {cum})")

# current saturated pool: medium + functional + post 2024-10
cur = [r for r in comp if r.get("difficulty") == "medium" and (r.get("contest_date") or "")[:10] >= "2024-10-01"]
print(f"\ncurrent saturated pool (medium, post-2024-10): {len(cur)}")

# candidate windows for 'not too new, not too hard'
for lo, hi, diffs in [("2024-01-01", "2025-02-01", ("medium",)),
                      ("2024-06-01", "2025-03-01", ("medium",)),
                      ("2024-01-01", "2025-02-01", ("easy", "medium")),
                      ("2023-09-01", "2025-02-01", ("medium",))]:
    pool = [r for r in comp if r.get("difficulty") in diffs
            and lo <= (r.get("contest_date") or "")[:10] < hi]
    print(f"window [{lo},{hi}) diffs={diffs}: {len(pool)} problems")
