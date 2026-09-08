#!/usr/bin/env python
"""Is armI's MPE loss a CAPABILITY loss or a REGISTER failure? Sweep every arm.

armI's extra MPE failures are ~all SyntaxError, with AssertionError flat -- when it emits
code, the code is as right as armD's. The new failures are (a) EMPTY and (b) chain-of-thought
PROSE leaking into a raw-completion stream ("The user wants a Java function...").

MPE is raw-completion: continue a function body as bare code, stop at "\\n}". A model that
slips into chat/thinking register produces exactly this signature. If that is the mechanism
it should be DOSE-ORDERED across the hybrid arms and absent from the non-hybrid ones -- and
it should NOT appear on the chat-mode benches (HE+ / LCB), where armI is fine or better.

Prints, per arm per language: empty rate, thinking-register rate, and the AssertionError
count (the genuine-wrong-logic channel) so capability and formatting are never conflated.
"""
import collections
import glob
import json
import os
import re

R = "/srv/ml/eval_results/ream_arms/multipl_e_100"
LANGS = {"humaneval-rs": "rs", "humaneval-java": "java", "humaneval-js": "js"}
ARMS = [
    ("pub184e", "pub184e_imat"),
    ("armD", "reamD_ourssal_nomerge"),
    ("armD_rpt", "reamD_rpt"),
    ("armE", "reamE_reapsal_nomerge"),
    ("armB", "reamB_reapsal_merge"),
    ("base256e", "base256e_imat"),
    ("armF rnorm", "reamF_rnorm_nomerge"),
    ("armI p12", "hybrid_p12_ourssal_reapfloor"),
    ("armJ p24", "hybrid_p24_ourssal_reapfloor"),
    ("armG gs2", "reamG_ourssal_merge_gs2"),
    ("armH gs4", "reamH_ourssal_merge_gs4"),
]

# markers of chat / chain-of-thought register appearing inside a BARE CODE stream
THINK = re.compile(
    r"(the user wants|here'?s a thinking|thinking process|let'?s (think|start|analyze)|"
    r"\*\*understand|we need to|the (function|problem|examples?) (is|are|shows?)|"
    r"i will |we can |step \d|first,|note that|explanation:|approach:)", re.I)
MD = re.compile(r"(^|\n)\s*(\d+\.\s|\*\*|- \*\*|#{1,4}\s)")


def samples(arm):
    p = os.path.join(R, arm, "mpe_result.samples.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("task_id") or d.get("doc_id")
            if t:
                out[t] = d.get("completion") or ""
    return out


def statuses(arm):
    out = {}
    for ldir, lang in LANGS.items():
        for p in glob.glob(os.path.join(R, arm, "results", ldir, "*.results.json")):
            if p.endswith("_summary.json"):
                continue
            try:
                d = json.load(open(p))
            except Exception:
                continue
            res = d.get("results") or []
            if res:
                out[(lang, d.get("name"))] = res[0].get("status", "?")
    return out


print("%-11s %-5s | %5s %6s %6s %6s | %s" %
      ("arm", "lang", "OK", "empty", "think", "assert", "note"))
print("-" * 78)
for label, arm in ARMS:
    S, ST = samples(arm), statuses(arm)
    if not S:
        continue
    for lang in ("rs", "java", "js"):
        keys = [(l, n) for (l, n) in ST if l == lang]
        if not keys:
            continue
        ok = sum(1 for k in keys if ST[k] == "OK")
        asrt = sum(1 for k in keys if "Assertion" in str(ST[k]))
        empty = think = 0
        for l, n in keys:
            c = S.get("%s::%s" % (l, n), "")
            if not c.strip():
                empty += 1
            elif THINK.search(c[:400]) or MD.search(c[:200]):
                think += 1
        print("%-11s %-5s | %5d %6d %6d %6d |" % (label, lang, ok, empty, think, asrt))
    print()
