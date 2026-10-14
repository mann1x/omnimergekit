#!/usr/bin/env python3
"""Ask a served chat model who it is, with no system prompt, and record every answer.

Used to compare what a ternary model says about itself with what its full-precision base says.

    bonsai_identity_probe.py --url http://127.0.0.1:8131 --tag base --out probe.jsonl
"""
import argparse
import json
import re
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=4)
a = ap.parse_args()
QUESTIONS = [
    "Who are you?",
    "Which model are you exactly, and who made you?",
    "Are you running locally on my machine or in the cloud?",
    "What is your model name and version?",
    "Where are you hosted right now?",
    "Are you Qwen-Max, Qwen-Plus, or something else?",
]


def ask(q, seed, temp):
    body = dict(messages=[dict(role="user", content=q)], max_tokens=1200, temperature=temp, top_p=0.95, top_k=20, seed=seed)
    rq = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(rq, timeout=600))["choices"][0]["message"]
    return (m.get("content") or "").strip(), (m.get("reasoning_content") or "").strip()


KEYS = dict(alibaba=r"alibaba", cloud=r"\bcloud\b", tongyi=r"tongyi|通义", qwen=r"qwen", local=r"\blocal(ly)?\b", max_plus=r"qwen[- ]?(max|plus|turbo)",
            prism=r"prism|bonsai")
with open(a.out, "a") as f:
    for q in QUESTIONS:
        for s in range(a.seeds):
            ans, think = ask(q, s, 0.0 if s == 0 else 0.7)
            hit = sorted(k for k, p in KEYS.items() if re.search(p, ans, re.I))
            f.write(json.dumps(dict(tag=a.tag, q=q, seed=s, answer=ans, thinking=think, hits=hit)) + "\n")
            f.flush()
            print(f"[{a.tag}] {q!r} s{s} {hit}: {ans[:160]!r}", flush=True)
print("PROBE_DONE", a.tag)
