#!/usr/bin/env python3
"""Stack Fix-A + reasoning-sidecar on stock lm_eval LocalChatCompletion.parse_generations."""
import sys
from pathlib import Path

SENTINEL = "reasoning-sidecar 2026-05-29"
STOCK = (
    "                for choices in out[\"choices\"]:\n"
    "                    tmp[choices[\"index\"]] = choices[\"message\"][\"content\"]"
)
NEW = (
    "                for choices in out[\"choices\"]:\n"
    "                    # Fix-A + reasoning-sidecar 2026-05-29 (stacks Fix-A)\n"
    "                    msg = choices[\"message\"]\n"
    "                    content = msg.get(\"content\") or \"\"\n"
    "                    if content:\n"
    "                        text = content\n"
    "                    else:\n"
    "                        text = msg.get(\"reasoning\") or msg.get(\"reasoning_content\") or \"\"\n"
    "                    import os as _os, json as _json\n"
    "                    _rlog = _os.environ.get(\"LM_EVAL_REASONING_LOG\")\n"
    "                    if _rlog:\n"
    "                        _rc = msg.get(\"reasoning_content\") or msg.get(\"reasoning\") or \"\"\n"
    "                        try:\n"
    "                            with open(_rlog, \"a\") as _f:\n"
    "                                _f.write(_json.dumps({\"idx\": choices[\"index\"], \"content_chars\": len(content), \"reasoning_chars\": len(_rc)}) + \"\\n\")\n"
    "                        except Exception:\n"
    "                            pass\n"
    "                    tmp[choices[\"index\"]] = text"
)

p = Path(sys.argv[1])
src = p.read_text()
if SENTINEL in src:
    print("skip-already-applied"); sys.exit(0)
if STOCK not in src:
    print(f"FAIL: stock anchor not in {p} (file modified?)"); sys.exit(1)
p.write_text(src.replace(STOCK, NEW, 1))
print("ok-patched")
