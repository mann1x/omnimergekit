import json, os, glob, hashlib
ROOTS = ["/mnt/sdc/v7rework/arms", "/mnt/sdc/ream-work", "/mnt/sdc/ml/models", "/srv/ml/models"]
KEYS = ["chat_template","added_tokens_decoder","additional_special_tokens","add_bos_token","extra_special_tokens"]
def sha(b): return hashlib.sha256(b).hexdigest()[:12]
rows=[]; seen=set()
for r in ROOTS:
    for tc in sorted(glob.glob(os.path.join(r,"*","tokenizer_config.json"))):
        d=os.path.dirname(tc)
        if d in seen: continue
        seen.add(d)
        raw=open(tc,"rb").read(); j=json.loads(raw)
        at=j.get("added_tokens_decoder") or {}
        jinja=os.path.join(d,"chat_template.jinja")
        rows.append(dict(dir=d, bytes=len(raw), atd=len(at),
            has_ct=("chat_template" in j),
            missing=[k for k in KEYS if k not in j],
            jinja=(sha(open(jinja,"rb").read()) if os.path.exists(jinja) else "-"),
            jinja_bytes=(os.path.getsize(jinja) if os.path.exists(jinja) else 0)))
hdr = "{:<44s} {:>9s} {:>4s} {:>6s} {:>12s} {:>8s}  {}"
print(hdr.format("DIR","tok_cfg B","atd","emb_ct","jinja_sha","jinja_B","missing_keys"))
for r in sorted(rows, key=lambda x: x["dir"]):
    flag = "   <== STRIPPED" if r["atd"] == 0 else ""
    print(hdr.format(r["dir"].replace("/mnt/sdc/",""), str(r["bytes"]), str(r["atd"]),
                     str(r["has_ct"]), r["jinja"], str(r["jinja_bytes"]),
                     (",".join(r["missing"]) or "-") + flag))
