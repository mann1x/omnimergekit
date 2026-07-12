#!/usr/bin/env python3
"""tri130_replay.py — replay the EXACT HumanEval/130 request that was sent to
dern11-Q4, against an arbitrary llama-server (here: dern11-F16), greedy, so only
precision differs. Reports rumination length (content + reasoning) and pass/fail.

Usage: tri130_replay.py <server_base_url>   e.g. http://127.0.0.1:8194
"""
import json
import re
import sys

import requests

SERVER = sys.argv[1].rstrip("/")
SAMPLES = ("/srv/ml/eval_results_dern11_tradecheck/humanevalplus_full/"
           "dern11-q4/lm_eval_out/dern11-q4/"
           "samples_humaneval_plus_chat_2026-06-17T08-29-45.351520.jsonl")

# pull the exact request + the test harness from the Q4 samples record
doc_test = entry = None
content = None
gen = None
for line in open(SAMPLES):
    r = json.loads(line)
    if r["doc"]["task_id"] == "HumanEval/130":
        a = r["arguments"]["gen_args_0"]
        msgs = json.loads(a["arg_0"][0])           # the [{"role":..,"content":..}] list
        content = msgs[0]["content"]
        gen = a["arg_1"]
        doc_test = r["doc"]["test"]
        entry = r["doc"]["entry_point"]
        break
assert content is not None, "tri-130 not found"

payload = {
    "messages": [{"role": "user", "content": content}],
    "temperature": 0.0, "top_p": 1.0, "top_k": 0,
    "max_tokens": gen["max_gen_toks"],
    "stop": gen["until"],
}
print("POST %s/v1/chat/completions  (max_tokens=%d, greedy)" % (SERVER, gen["max_gen_toks"]))
resp = requests.post(SERVER + "/v1/chat/completions", json=payload, timeout=1500).json()
ch = resp["choices"][0]
msg = ch["message"]
cont = msg.get("content") or ""
reason = msg.get("reasoning_content") or ""
print("finish_reason =", ch.get("finish_reason"))
print("content_len   =", len(cont))
print("reasoning_len =", len(reason))
print("=" * 70)
print("CONTENT:")
print(cont)
print("=" * 70)

# extract the python block and run the canonical test harness
m = re.search(r"```python\s*(.*?)```", cont, re.DOTALL)
code = m.group(1) if m else cont
verdict = "UNKNOWN"
try:
    ns = {}
    exec(code, ns)
    exec(doc_test, ns)
    ns["check"](ns[entry])
    verdict = "PASS"
except Exception as e:
    verdict = "FAIL (%s: %s)" % (type(e).__name__, str(e)[:200])
print("TRI-130 VERDICT:", verdict)
