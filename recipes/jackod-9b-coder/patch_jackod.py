import json, sys, shutil, os
q, o = sys.argv[1], sys.argv[2]
c = json.load(open(os.path.join(o, "config.json")))
qc = json.load(open(os.path.join(q, "config.json")))
for k in ("eos_token_id", "pad_token_id"):
    if qc.get(k) is not None:
        c[k] = qc[k]
json.dump(c, open(os.path.join(o, "config.json"), "w"), indent=2)
print("  root eos now:", c.get("eos_token_id"))
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
    s = os.path.join(q, f)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(o, f))
        print("  copied", f)
