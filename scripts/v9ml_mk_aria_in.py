"""Write the aria2c input file (0600) for the pinned 128e revision; token never on argv."""
import json
import os

from huggingface_hub import get_token

D = "/mnt/sdc/ml/models/gemma-4-26B-A4B-it"
IN = "/mnt/sdc/ml/models/.aria2_gemma128e.in"
tok = get_token()
if not tok:
    raise SystemExit("no HF token")
ref = json.load(open(f"{D}/LFS_SHA256.json"))
rev = ref["revision"]
old = os.umask(0o077)
with open(IN, "w") as f:
    for fn in ref["sha256"]:
        url = f"https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/{rev}/{fn}"
        f.write(f"{url}\n  out={fn}\n  dir={D}\n  header=Authorization: Bearer {tok}\n")
os.umask(old)
os.chmod(IN, 0o600)
print("input file written for", len(ref["sha256"]), "shards")
