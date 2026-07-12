#!/usr/bin/env python3
"""he_diff_dern11.py — diff HE+ per-problem pass/fail between dern11 and noswap,
dump the generations for the disagreements so we can diagnose WHY dern11 failed
the ones noswap passed. Usage: he_diff_dern11.py <dern11.jsonl> <noswap.jsonl>"""
import json
import sys


def load(p):
    d = {}
    for line in open(p):
        r = json.loads(line)
        tid = r["doc"]["task_id"]
        gen = r.get("filtered_resps") or r.get("resps") or [""]
        while isinstance(gen, list):
            gen = gen[0] if gen else ""
        raw = r.get("resps") or [""]
        while isinstance(raw, list):
            raw = raw[0] if raw else ""
        d[tid] = {"pass": float(r.get("pass@1", 0.0)), "gen": gen, "raw_len": len(raw),
                  "entry": r["doc"].get("entry_point"), "prompt": r["doc"].get("prompt", "")}
    return d


A = load(sys.argv[1])   # dern11
B = load(sys.argv[2])   # noswap
da = sum(1 for t in A if A[t]["pass"] < 1)
db = sum(1 for t in B if B[t]["pass"] < 1)
dern_only = sorted(t for t in A if A[t]["pass"] < 1 and B.get(t, {}).get("pass", 0) >= 1)
noswap_only = sorted(t for t in A if A[t]["pass"] >= 1 and B.get(t, {}).get("pass", 0) < 1)
both_fail = sorted(t for t in A if A[t]["pass"] < 1 and B.get(t, {}).get("pass", 0) < 1)
print("dern11 total fails: %d/164   noswap total fails: %d/164" % (da, db))
print("DERN-ONLY fails (dern FAIL, noswap PASS): %s" % dern_only)
print("NOSWAP-ONLY fails (dern PASS, noswap FAIL): %s" % noswap_only)
print("BOTH fail: %s" % both_fail)
print()
for t in dern_only:
    print("=" * 90)
    print("TASK %s   entry_point=%s" % (t, A[t]["entry"]))
    print("--- PROMPT (signature+docstring) ---")
    print(A[t]["prompt"])
    print("--- DERN11 extracted code (FAILED)   [raw_resp_len=%d] ---" % A[t]["raw_len"])
    print(A[t]["gen"])
    print("--- NOSWAP extracted code (PASSED)   [raw_resp_len=%d] ---" % B[t]["raw_len"])
    print(B[t]["gen"])
    print()
