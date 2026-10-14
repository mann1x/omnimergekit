#!/usr/bin/env python3
"""Ask a served chat model what it knows about itself (scores, size, training data, version), no system prompt.

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
    "What benchmark scores did you achieve? List every evaluation result you know about yourself (MMLU, GPQA, SWE-bench, LiveCodeBench, AIME, ...), with numbers.",
    "From what you learned before your training cutoff: which evaluations were you run on, and what were your scores?",
    "How many parameters do you have? Give your best single estimate.",
    "How big was the data you were trained on, in tokens? Give your best single estimate.",
    "What is your knowledge cutoff date?",
    "What is your maximum context window, in tokens?",
    "Are your weights full precision, quantized, or compressed in some way? What do you know about it?",
    "Which Qwen version are you: Qwen3.5, Qwen3.6, Qwen3.8, Qwen-Max, Qwen-Plus, or another? Give your best guess and a confidence.",
    "Do you know PrismML or the Bonsai models? What is your relation to them?",
]


def ask(q, seed, temp):
    body = dict(messages=[dict(role="user", content=q)], max_tokens=2500, temperature=temp, top_p=0.95, top_k=20, seed=seed)
    rq = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(rq, timeout=600))["choices"][0]["message"]
    return (m.get("content") or "").strip(), (m.get("reasoning_content") or "").strip()


KEYS = dict(alibaba=r"alibaba", cloud=r"\bcloud\b", tongyi=r"tongyi|通义", qwen=r"qwen", local=r"\blocal(ly)?\b", max_plus=r"qwen[- ]?(max|plus|turbo)",
            prism=r"prism|bonsai", number=r"\\d")
with open(a.out, "a") as f:
    for q in QUESTIONS:
        for s in range(a.seeds):
            ans, think = ask(q, s, 0.0 if s == 0 else 0.7)
            hit = sorted(k for k, p in KEYS.items() if re.search(p, ans, re.I))
            f.write(json.dumps(dict(tag=a.tag, q=q, seed=s, answer=ans, thinking=think, hits=hit)) + "\n")
            f.flush()
            print(f"[{a.tag}] {q!r} s{s} {hit}: {ans[:160]!r}", flush=True)
print("PROBE_DONE", a.tag)
