"""CPU smoke for router_diff_v9fail.py: render, judge, and the 128e router-hook contract."""
import json
import sqlite3  # noqa: F401
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
import router_diff_v9fail as R  # noqa: E402
from datasets import load_dataset  # noqa: E402
from lm_eval.tasks.ifeval.utils import process_results  # noqa: E402

tok = AutoTokenizer.from_pretrained("/mnt/sdc/ml/models/gemma-4-26B-A4B-it")
s = R.render(tok, "Say hi in Kannada only.")
print("render tail:", repr(s[-80:]))
docs = {d["key"]: d for d in load_dataset("google/IFEval", split="train")}
rows = [json.loads(x) for x in open("/mnt/sdc/ml/corpora/v9ml/ifeval_calib_v9ml.jsonl")][:2]
for r in rows:
    print("judge ifeval:", R.judge(r, "Hello, this is an answer.", docs, process_results))
print("judge kn good:", R.judge({"src": "aya", "lang": "Kannada"}, "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಿ?", docs, process_results))
print("judge kn mixed:", R.judge({"src": "aya", "lang": "Kannada"}, "ಹೌंदी, ಹamburger सاندဝিች (sandwich) 입니다.", docs, process_results))
m = AutoModelForCausalLM.from_pretrained("/mnt/sdc/ml/models/gemma-4-26B-A4B-it", dtype=torch.bfloat16,
                                         low_cpu_mem_usage=True, device_map={"": "cpu"}).eval()
routers = [(n, mod) for n, mod in m.named_modules() if n.endswith(".router")]
print("routers:", len(routers), routers[0][0], "E=", m.config.text_config.num_experts)
cap = {}
routers[0][1].register_forward_hook(lambda _m, _i, out: cap.setdefault("o", out))
ids = tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"]
with torch.no_grad():
    m(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
o = cap["o"]
print("hook out type:", type(o).__name__, [tuple(x.shape) for x in (o if isinstance(o, tuple) else (o,))])
p = (o[0] if isinstance(o, tuple) else o).float()
p = p[0] if p.dim() == 3 else p
print("first-elem shape", tuple(p.shape), "row sums", p.sum(-1)[:4].tolist())
print("SMOKE_OK")
