#!/usr/bin/env python3
"""Pin the REAM quant chain to GPU1.

bs2 GPU1 is the device allocated to this work; GPU0 is not mine. quant_arms_imat.sh only
POLLED GPU1 for free VRAM and never exported CUDA_VISIBLE_DEVICES. quantize_gguf.py honors
that variable when set, but when unset it sums every physical GPU and llama-imatrix at
-ngl 99 spreads across all visible devices -- so the imatrix phase could and probably did
run on GPU0 too. Polling a GPU is not reserving it.

Idempotent: refuses to double-patch.
"""
import sys

P = "/mnt/sdc/ream-work/quant_arms_imat.sh"
ANCHOR = 'THREADS=$(( $(nproc) / 2 )); [ "$THREADS" -ge 4 ] || THREADS=4\n'
ADD = ANCHOR + """
# GPU1 is the ONLY device this work may touch; GPU0 belongs to someone else. Polling GPU1
# for free VRAM (below) is not reserving it: unset CUDA_VISIBLE_DEVICES lets quantize_gguf
# sum every physical GPU and lets llama-imatrix offload across both. Pin, do not poll.
export CUDA_VISIBLE_DEVICES=1
"""

s = open(P).read()
if "CUDA_VISIBLE_DEVICES" in s:
    print("already pinned -- no change")
    sys.exit(0)
if ANCHOR not in s:
    sys.exit("FAIL: anchor line not found; refusing to guess an insertion point")
open(P, "w").write(s.replace(ANCHOR, ADD, 1))

# artifact gate: re-read from disk
back = open(P).read()
if "export CUDA_VISIBLE_DEVICES=1" not in back:
    sys.exit("FAIL: pin absent after write")
print("pinned: export CUDA_VISIBLE_DEVICES=1")
