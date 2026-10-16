#!/usr/bin/env python3
"""Per-prompt table of the v9-ml quick test (ifeval_v9fail10) across arms, with a script census."""
import collections
import glob
import json
import unicodedata

R = "/mnt/sdc/ml/v9ml/quick/results/ifeval_v9fail10"
ARMS = ["A", "shared_k2", "shared_k4", "shared_k6", "shared_k8", "ml_k8", "if_k8"]
rows, lang = {}, {}
for a in ARMS:
    f = glob.glob(f"{R}/v9ml-{a}-q6k/lm_eval_out/*/samples_*.jsonl")[0]
    s = json.load(open(f"{R}/v9ml-{a}-q6k/summary.json"))
    for line in open(f):
        d = json.loads(line)
        doc = d["doc"]
        r = d["resps"][0][0] if isinstance(d["resps"][0], list) else d["resps"][0]
        k = doc["key"]
        if "language:response_language" in doc["instruction_id_list"]:
            lang[k] = [kw.get("language") for kw in doc["kwargs"] if kw and kw.get("language")]
        rows.setdefault(k, {})[a] = (bool(d.get("prompt_level_strict_acc")), len(r), r)
    print(f"{a:10s} score={s.get('score')} metric={s.get('metric')} sampler={(s.get('sampler') or {}).get('name')}")


def script(t):
    c = collections.Counter(unicodedata.name(ch, "?").split()[0] for ch in t if ch.isalpha())
    tot = sum(c.values()) or 1
    return ",".join(f"{k}{v * 100 // tot}" for k, v in c.most_common(2))


print("\nkey    " + " ".join(f"{a:>10s}" for a in ARMS))
for k in sorted(rows, key=lambda x: (x not in lang, x)):
    tag = f"{k}{'*' if k in lang else ' '}"
    print(f"{tag:7s}" + " ".join(f"{('PASS' if rows[k][a][0] else 'fail') + '/' + str(rows[k][a][1] // 100):>10s}" for a in ARMS))
print("\n* response_language:", lang, "| cell = verdict/length in 100s of chars")
for k in lang:
    print(f"\n{k} script share (top-2):")
    for a in ARMS:
        print(f"   {a:10s} {script(rows[k][a][2]):30s} {rows[k][a][2][:70]!r}")
print("\nempty/<5ch cells:", sum(1 for k in rows for a in ARMS if rows[k][a][1] < 5))
