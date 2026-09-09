from transformers import AutoTokenizer
import json, os
ARM="/mnt/sdc/v7rework/arms/Jprime-p3-bf16"
BASE=None
for c in ["/mnt/sdc/ml/models/gemma-4-26B-A4B-it","/srv/ml/models/gemma-4-26B-A4B-it"]:
    if os.path.isdir(c): BASE=c; break
for name,P in [("ARM Jprime-p3",ARM),("BASE gemma-4-it",BASE)]:
    if P is None: print(f"### {name}: NOT FOUND locally"); continue
    t=AutoTokenizer.from_pretrained(P, trust_remote_code=True)
    print(f"### {name}  ({type(t).__name__}, vocab {len(t)})")
    for i in [1,3,50,105,106,107]:
        print(f"    id {i:4d} -> {t.convert_ids_to_tokens(i)!r}")
    print("    eos_token:", repr(t.eos_token), "id", t.eos_token_id)
    gc=os.path.join(P,"generation_config.json")
    if os.path.exists(gc):
        g=json.load(open(gc)); print("    generation_config eos_token_id:", g.get("eos_token_id"))
    tc=json.load(open(os.path.join(P,"tokenizer_config.json")))
    at=tc.get("added_tokens_decoder") or {}
    print("    added_tokens_decoder entries:", len(at))
