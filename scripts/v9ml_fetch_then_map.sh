#!/usr/bin/env bash
# Fetch the 128e bf16 (revision pinned in LFS_SHA256.json), verify LFS sha256, then run the map.
set -u
D=/mnt/sdc/ml/models/gemma-4-26B-A4B-it
IN=/mnt/sdc/ml/models/.aria2_gemma128e.in
echo "[fetch] start $(date -u +%FT%TZ)"
aria2c --disable-ipv6=true -x8 -s8 -j2 --continue=true --file-allocation=none --console-log-level=warn --summary-interval=60 -i "$IN"
arc=$?
rm -f "$IN"
echo "[fetch] aria2c rc=$arc, token input file removed $(date -u +%FT%TZ)"
[ $arc -eq 0 ] || { echo "V9ML_FETCH_FAIL aria2c"; exit 2; }
/srv/ml/envs/envs/omnimergekit/bin/python - "$D" <<'PY' || { echo "V9ML_FETCH_FAIL sha"; exit 3; }
import hashlib, json, sys
d = sys.argv[1]; ref = json.load(open(f"{d}/LFS_SHA256.json"))["sha256"]
bad = 0
for fn, want in ref.items():
    h = hashlib.sha256()
    with open(f"{d}/{fn}", "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    ok = h.hexdigest() == want; bad += not ok
    print(f"[sha] {fn} {OK if ok else MISMATCH} {h.hexdigest()[:16]}", flush=True)
sys.exit(1 if bad else 0)
PY
echo "[fetch] V9ML_FETCH_OK $(date -u +%FT%TZ)"
bash /mnt/sdc/harness/v9ml_map.sh
