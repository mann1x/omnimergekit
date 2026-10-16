#!/usr/bin/env python3
"""Re-upload tiers whose quantize_gguf upload gave up (HF storage quota, 2026-10-16), sha-gated.
For each tier: local sha256 == sidecar -> upload_file -> hub LFS sha == sidecar -> delete the local GGUF
(the sidecar stays; pod_v9_xollama_publish.py downloads from HF). Usage: pod_v9pub_reupload.py TIER [TIER ...]"""
import hashlib
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "ManniX-ITA/gemma-4-20B-v9-agentic-GGUF"
GG = Path("/workspace/v9pub/gguf/gemma-4-20B-v9-agentic")
api = HfApi()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


rc = 0
for tier in sys.argv[1:]:
    name = f"gemma-4-20B-v9-agentic-{tier}.gguf"
    p, side = GG / name, GG / f"{name}.sha256"
    want = side.read_text().split()[0]
    if not p.exists():
        print(f"{tier}: REUPLOAD_FAIL no local file", flush=True); rc = 1; continue
    got = sha256(p)
    if got != want:
        print(f"{tier}: REUPLOAD_FAIL local sha {got[:12]} != sidecar {want[:12]}", flush=True); rc = 1; continue
    for attempt in range(1, 6):
        try:
            t0 = time.time()
            api.upload_file(path_or_fileobj=str(p), path_in_repo=name, repo_id=REPO,
                            commit_message=f"Add {tier} quantization ({p.stat().st_size / 1e9:.2f} GB)")
            break
        except Exception as e:
            print(f"{tier}: upload attempt {attempt} failed: {str(e)[:200]}", flush=True)
            time.sleep(60 * attempt)
    else:
        print(f"{tier}: REUPLOAD_FAIL upload", flush=True); rc = 1; continue
    hub = {s.rfilename: (s.lfs.sha256 if s.lfs else None)
           for s in api.model_info(REPO, files_metadata=True).siblings}.get(name)
    if hub != want:
        print(f"{tier}: REUPLOAD_FAIL hub sha {hub} != sidecar {want[:12]} (local kept)", flush=True); rc = 1; continue
    p.unlink()
    print(f"{tier}: REUPLOAD_OK {time.time() - t0:.0f}s hub sha == sidecar {want[:12]}; local deleted, sidecar kept", flush=True)
sys.exit(rc)
