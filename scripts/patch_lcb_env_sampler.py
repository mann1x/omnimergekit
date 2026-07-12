#!/usr/bin/env python3
"""Additive, idempotent patch: make eval/lcb/lcb_llama_server.py read its
sampler from LCB_* env vars, defaulting to the current hardcoded greedy
(temperature 0.0 / top_p 1.0, no top_k/min_p/repeat_penalty). When the env
vars are set, run LCB under the served anti-loop sampler vendor_minp_rep —
same top-level JSON keys replay_harness forwards to llama-server.

Unset env => byte-identical behavior to today, so the canonical frozen-greedy
LCB path and any in-flight greedy run are unaffected.
"""
import ast
import sys

PATH = "/srv/ml/repos/omnimergekit/eval/lcb/lcb_llama_server.py"

ANCHOR = (
    '        "temperature": 0.0,\n'
    '        "top_p": 1.0,\n'
    '        "max_tokens": max_tokens,\n'
    '        "stream": False,\n'
    "    }\n"
)

REPLACEMENT = (
    '        "temperature": float(os.environ.get("LCB_TEMPERATURE", "0.0")),\n'
    '        "top_p": float(os.environ.get("LCB_TOP_P", "1.0")),\n'
    '        "max_tokens": max_tokens,\n'
    '        "stream": False,\n'
    "    }\n"
    "    # Optional deployment-sampler overrides (env-driven; default = canonical\n"
    "    # greedy, so unset env => byte-identical behavior). Used to run LCB under\n"
    "    # the served anti-loop sampler vendor_minp_rep (top_k 64 / min_p 0.05 /\n"
    "    # repeat_penalty 1.1 / top_p 0.95 / temp 0.9) — same top-level keys\n"
    "    # replay_harness forwards to llama-server.\n"
    '    _lcb_topk = os.environ.get("LCB_TOP_K")\n'
    "    if _lcb_topk:\n"
    '        payload["top_k"] = int(_lcb_topk)\n'
    '    _lcb_minp = os.environ.get("LCB_MIN_P")\n'
    "    if _lcb_minp:\n"
    '        payload["min_p"] = float(_lcb_minp)\n'
    '    _lcb_reppen = os.environ.get("LCB_REPEAT_PENALTY")\n'
    "    if _lcb_reppen:\n"
    '        payload["repeat_penalty"] = float(_lcb_reppen)\n'
)

src = open(PATH, encoding="utf-8").read()

if "LCB_TEMPERATURE" in src:
    print("ALREADY_PATCHED — no-op")
    sys.exit(0)

n = src.count(ANCHOR)
if n != 1:
    print(f"FATAL: anchor found {n} times (expected 1) in {PATH} — refusing to patch")
    sys.exit(3)

src2 = src.replace(ANCHOR, REPLACEMENT, 1)
# sanity: must still compile
ast.parse(src2)
open(PATH, "w", encoding="utf-8").write(src2)
print("PATCHED_OK — env-driven sampler added, greedy default preserved")
# echo the patched payload region for confirmation
i = src2.index("payload = {")
print("--- patched region ---")
print(src2[i:i + 900])
