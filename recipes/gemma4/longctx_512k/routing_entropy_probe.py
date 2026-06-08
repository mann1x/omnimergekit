#!/usr/bin/env python
# routing_entropy_probe.py — v6-coder MoE routing health check after YaRN
# extension. Captures router logits at specific positional buckets and
# computes per-layer Shannon entropy + per-expert dominance.
#
# Plan v2 §"Phase 2 step 6" + brief §5 §"Routing-entropy gate":
#  - For each of N docs (default 100, ≥4k tokens each), tokenize and pad/truncate
#    to 524288 tokens. Run forward pass capturing router_logits per layer
#    (Gemma 4 only fires router on full-attn-layer outputs — so 5 router events
#    per token in v6-coder).
#  - Bucket the captured logits by position bucket: {32k, 256k, 512k}.
#    32k bucket = positions [0, 32768).
#    256k bucket = positions [196608, 262144).
#    512k bucket = positions [491520, 524288).
#  - For each (layer, bucket) compute:
#      (a) entropy = Shannon entropy of the top-8 softmax over experts
#                    (Gemma 4 routes top-8 of 98 experts in v6-coder)
#      (b) dominance = max single-expert share of routed tokens
#      (c) utilization = fraction of 98 experts that received any tokens
#  - Gate (must hold to publish):
#      entropy_256k / entropy_32k >= 0.85    per layer
#      entropy_512k / entropy_32k >= 0.85    per layer
#      dominance @ 256k or 512k < 0.40       per layer
#  - Optional gate (council Q5c):
#      utilization @ 256k or 512k >= 0.80 * utilization @ 32k
#
# Output: <out>/routing_entropy.json with per-(layer, bucket) measurements +
# verdict per gate.
#
# ### COUNCIL — read brief §5 Q5b/c. Open: looser 0.75× threshold? Adding
# the utilization gate?
#
# Inputs:
#   --model       path to the merged 512k v6-coder dir
#   --positions   comma-list of buckets to probe (default 32k,256k,512k)
#   --n-docs      number of long docs to use (default 100)
#   --doc-source  source of long docs; default 'pg19' (HuggingFace)
#   --out         output dir (writes routing_entropy.json)
#
# ### Not yet implemented:
#   - hook registration on router.proj for each full-attn layer
#   - vLLM doesn't expose router logits at the API level — must use
#     transformers forward pass with output_router_logits=True (Gemma4MoE
#     supports this) — slow at 524288 ctx, expect ~5min/doc on Blackwell.
#     For n=100 docs, this is ~8h. Council may suggest reducing n=50.
#   - bucketing + entropy + dominance + utilization compute

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged v6-coder-512k dir")
    ap.add_argument("--positions", default="32k,256k,512k")
    ap.add_argument("--n-docs", type=int, default=100)
    ap.add_argument("--doc-source", default="pg19")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=== routing_entropy_probe ===")
    print(f"  model    : {args.model}")
    print(f"  positions: {args.positions}")
    print(f"  n_docs   : {args.n_docs}")
    print(f"  source   : {args.doc_source}")
    print(f"  out      : {args.out}")

    if args.dry_run:
        print("[dry-run] no-op.")
        return 0

    # TODO COUNCIL-APPROVED — implement.
    print("FATAL: routing_entropy_probe not yet implemented (council review pending).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
