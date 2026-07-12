#!/usr/bin/env python3
"""Probe: tokenized prompt length per agentic fixture + flash-attn availability.
Zero-GPU. Decides the OOM fix (flash_attention_2 vs context truncation)."""
import json
import os

from transformers import AutoTokenizer

LOOPER = "/mnt/sdc/ml/sft_heal/fkbroad-selection-looper-it"
FXD = "/srv/ml/agentic_loop/fixtures"
FX = ["solar_build_start", "threejs_build", "csharp_loop_parity"]

print("== flash_attn available? ==")
try:
    import flash_attn  # noqa: F401
    print("  flash_attn YES version", getattr(flash_attn, "__version__", "?"))
except Exception as e:  # noqa: BLE001
    print("  flash_attn NO (%s)" % type(e).__name__)

tok = AutoTokenizer.from_pretrained(LOOPER, trust_remote_code=True)
print("== tokenized prompt lengths (apply_chat_template + add_generation_prompt) ==")
for nm in FX:
    p = os.path.join(FXD, nm + ".json")
    fx = json.load(open(p))
    msgs = fx["messages"]
    tools = fx.get("tools")
    try:
        chat = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
        mode = "with-tools"
    except Exception as e:  # noqa: BLE001
        chat = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        mode = "no-tools(%s)" % type(e).__name__
    ids = tok(chat, add_special_tokens=False)["input_ids"]
    print("  %-22s plen=%6d tokens  (%s, n_msgs=%d)" % (nm, len(ids), mode, len(msgs)))
