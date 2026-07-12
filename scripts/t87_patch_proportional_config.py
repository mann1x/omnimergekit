#!/usr/bin/env python3
"""Patch a Gemma 4 `proportional_yarn` config -> `proportional` base rope for GGUF convert.

llama.cpp's gemma converter asserts `rope_type == 'proportional'` (gemma.py
generate_extra_tensors). The trained extension uses the CUSTOM `proportional_yarn`
(proportional base + YaRN ramp). YaRN is applied at llama-server RUNTIME
(`--rope-scaling yarn --yarn-orig-ctx 262144 --rope-scale 2.0 ...`), NOT baked into
the GGUF. So the GGUF must carry the *proportional base* rope (identical to how base
Gemma 4 GGUFs are produced); runtime composes the YaRN extension on top.

Usage: t87_patch_proportional_config.py <src_config.json> <out_config.json>
"""
import json
import sys

src, out = sys.argv[1], sys.argv[2]
c = json.load(open(src))
# YaRN-only fields to drop when collapsing proportional_yarn -> proportional
YK = ["factor", "original_max_position_embeddings", "beta_fast", "beta_slow",
      "mscale", "mscale_all_dim", "truncate"]


def walk(o):
    if isinstance(o, dict):
        if o.get("rope_type") == "proportional_yarn":
            o["rope_type"] = "proportional"
            for k in YK:
                o.pop(k, None)
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)


def normalize_mpe(o):
    # proportional base window is the YaRN original_max (262144); the extension
    # to 524288 is applied at runtime via yarn, so the GGUF declares the base ctx.
    if isinstance(o, dict):
        if o.get("max_position_embeddings") == 524288:
            o["max_position_embeddings"] = 262144
        for v in o.values():
            normalize_mpe(v)


walk(c)
normalize_mpe(c)
json.dump(c, open(out, "w"), indent=2)
print("patched: proportional_yarn -> proportional; max_position_embeddings 524288 -> 262144")
