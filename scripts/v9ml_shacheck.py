import hashlib
import json
import sys

d = "/mnt/sdc/ml/models/gemma-4-26B-A4B-it"
ref = json.load(open(f"{d}/LFS_SHA256.json"))["sha256"]
bad = 0
for fn, want in ref.items():
    h = hashlib.sha256()
    with open(f"{d}/{fn}", "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    verdict = "OK" if h.hexdigest() == want else "MISMATCH"
    bad += verdict != "OK"
    print(f"[sha] {fn} {verdict} {h.hexdigest()[:16]}", flush=True)
sys.exit(1 if bad else 0)
