import json, sys, urllib.request
from transformers import AutoTokenizer

MODEL = "/mnt/sdc/ream-work/armJ"
URL = "http://127.0.0.1:8477/generate/"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
msgs = [{"role": "user", "content": "Write a Python function that returns the sum of a list of integers. Reply with code only."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("PROMPT_REPR:", repr(prompt[-300:]), flush=True)

body = {"prompts": [prompt], "n": 1,
        "generation_kwargs": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256}}
req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    out = json.loads(r.read())
ids = out["completion_ids"][0]
print("N_TOK:", len(ids), flush=True)
print("TEXT_START>>>")
print(tok.decode(ids, skip_special_tokens=False))
print("<<<TEXT_END")
