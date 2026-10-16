#!/usr/bin/env python3
"""Run quantize_gguf.sanity_check on tiers whose quantize_gguf upload gave up (HF quota, 2026-10-16)
and were re-uploaded by pod_v9pub_reupload.py -- their sanity never ran. Each tier is downloaded from
the hub, sha-gated against this run's sidecar, checked with the SAME function quantize_gguf uses, then
removed. Prints quantize_gguf's own "  TIER: sanity PASS|FAIL (n/3)" lines.
Usage: CUDA_VISIBLE_DEVICES=1 pod_v9pub_resanity.py TIER [TIER ...]"""
import hashlib
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, "/workspace/omnimergekit/scripts")
import quantize_gguf as qg  # noqa: E402

REPO = "ManniX-ITA/gemma-4-20B-v9-agentic-GGUF"
GG = Path("/workspace/v9pub/gguf/gemma-4-20B-v9-agentic")
DL = Path("/workspace/v9pub/resanity")
tools = qg.find_llama_cpp()
print("tools:", {k: str(v) for k, v in tools.items() if k in ("server", "quantize")}, flush=True)
rc = 0
for tier in sys.argv[1:]:
    name = f"gemma-4-20B-v9-agentic-{tier}.gguf"
    want = (GG / f"{name}.sha256").read_text().split()[0]
    p = Path(hf_hub_download(REPO, name, local_dir=DL))
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    if h.hexdigest() != want:
        print(f"RESANITY_FAIL {tier}: downloaded sha {h.hexdigest()[:12]} != sidecar {want[:12]}", flush=True)
        rc = 1
        continue
    ok = qg.sanity_check(tools, p, tier, port=18199)
    print(f"RESANITY {tier} sha {want[:12]} -> {'PASS' if ok else 'FAIL'}", flush=True)
    rc |= 0 if ok else 1
    p.unlink()
sys.exit(rc)
